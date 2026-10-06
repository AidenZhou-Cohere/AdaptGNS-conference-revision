#!/usr/bin/env python3
"""Prospective saved Goop batches and bounded same-CUDA numerical checks only.

No CPU/CUDA parity claim, trajectory selection by outcomes, test access,
scientific checkpoint promotion, automatic retry or process launch.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = 'adaptgns_goop_graph_support_cuda_validation_v1'
BATCH_SCHEMA = 'adaptgns_goop_prospective_training_batches_v1'
SELECTION_RELEASE = 'adaptgns_goop_training_batch_selection_release_v1'
VALIDATION_RELEASE = 'adaptgns_goop_graph_support_numerical_release_v1'
TRAINER_SHA = 'dd9ef01a16116f04a578bbfe9ed01774e66fdc08d60b913072a33ad211b319c1'
ADMISSION_SHA = 'c2a12ef0c55b47f4c9493027450f8b51f52dbbd6043915bb09c1648b6cb9edeb'
PINS = {
    'train_goop_graph_support_cuda.py': TRAINER_SHA,
    'train_sand_graph_support_cuda.py': 'fb3053c892a7b5006178586f35617518aa3ca0df814a9f4247d647b83e8dc124',
    'validate_sand_graph_support_cuda.py': '365265ed25d60e1ebe50da89ec191b1a75a4773d2c75037952a756db3ec40e84',
    'validate_cuda_execution.py': '21b2555e84a2c6221b0651c5bcc592b67a61ae5b7e3dd02da031054dd79e2a68',
    'validate_sand_cuda.py': '60083ba485bf0e99e9fef34b9c76d81a0ad109d517974d5ef25657203fb1081c',
}
CASES = ('small', 'median', 'large')
BRANCHES = ('old_native', 'base', 'mix')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def private_import(directory, name):
    path = Path(directory)/name
    require(sha(path) == PINS[name], 'Pinned preparation helper differs: '+name)
    spec = importlib.util.spec_from_file_location('_private_goop_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    directory = Path(__file__).resolve().parent
    return {name: private_import(directory, name) for name in PINS}


def shared_numeric_check():
    directory = Path(__file__).resolve().parent
    paths = [directory/name for name in ('train_sand_graph_support_cuda.py', 'train_goop_graph_support_cuda.py')]
    sources = [path.read_text() for path in paths]
    nodes = [{node.name: node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
             for source in sources]
    names = ('configure_cuda', 'guarded_update', 'assert_adam', 'capture_rng', 'restore_rng', 'tree_equal',
             'graph_exposure_config', 'validate_graph_history')
    if any(ast.dump(nodes[0][name]) != ast.dump(nodes[1][name]) for name in names):
        return False
    old = ast.get_source_segment(sources[0], nodes[0]['run_training']).replace('sand', 'goop').replace('Sand', 'Goop')
    new = ast.get_source_segment(sources[1], nodes[1]['run_training'])
    new = new.replace('        reverify_goop_evidence_and_auxiliaries(args, helpers)\n', '')
    return ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))


def describe():
    return {'schema': SCHEMA, 'status': 'description_only', 'scientific_training_admitted': False,
            'preparation': 'separate root-released three-batch save before CUDA outcomes',
            'schedule': 'sort complete1000 train trajectories by(particle count,source index); pairs ranks0/1,500/501,998/999; targets6,200',
            'seed': 0, 'cases': list(CASES), 'branches': list(BRANCHES),
            'native_reference': 'pinned original forward path on same Goop initial state/device; no earlier trained model',
            'max_optimizer_updates': 15, 'replay': 'base and mix small case; first update, saved checkpoint, uninterrupted/resumed second update',
            'cpu_cuda_comparison': 'none; failed SandCPU/CUDA comparisons remain separate', 'source_pins': PINS}


def expected_schedule(manifest, selector):
    require(manifest.get('dataset') == 'Goop' and manifest.get('split') == 'train'
            and manifest.get('record_count') == len(manifest.get('records', [])) == 1000,
            'Complete Goop training manifest required for prospective schedule')
    require(all(r['positions']['shape'][0] == 401 for r in manifest['records']), 'Fixed401 frames required')
    schedule = selector.case_schedule(manifest['records'])
    require([r['case'] for r in schedule] == list(CASES)
            and all(r['target_frames'] == [6, 200] for r in schedule), 'Prospective case schedule differs')
    return schedule


def validate_selection_release(release, trainer, args, schedule):
    require(release.get('schema') == SELECTION_RELEASE and release.get('status') == 'admitted_for_batch_preparation'
            and release.get('issued_by') == 'root' and release.get('scientific_training_admitted') is False
            and release.get('train_manifest_sha256') == trainer.MANIFEST_SHA
            and release.get('training_admission_sha256') == sha(args.admission) == ADMISSION_SHA
            and release.get('trainer_sha256') == TRAINER_SHA and release.get('preparation_source_sha256') == sha(__file__)
            and release.get('protocol_sha256') == sha(args.protocol) and release.get('schedule') == schedule,
            'Exact prospective root batch-selection release required')


def prepare_batches(args, mods):
    trainer, selector = mods['train_goop_graph_support_cuda.py'], mods['validate_sand_cuda.py']
    require(sha(args.train_manifest) == trainer.MANIFEST_SHA, 'Pinned Goop train manifest required')
    manifest = read(args.train_manifest); schedule = expected_schedule(manifest, selector)
    validate_selection_release(read(args.selection_release), trainer, args, schedule)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    started = time.perf_counter()
    report = {'schema': BATCH_SCHEMA, 'status': 'preparing', 'cases': [], 'schedule': schedule,
              'source_sha256': sha(__file__), 'helper_source_sha256': PINS,
              'train_manifest_sha256': trainer.MANIFEST_SHA, 'training_admission_sha256': sha(args.admission),
              'selection_release_sha256': sha(args.selection_release), 'protocol_sha256': sha(args.protocol),
              'metadata_sha256': trainer.METADATA_SHA, 'scientific_training_admitted': False,
              'selection_uses_outcomes': False, 'new_optimizer_updates': 0, 'test_accessed': False}
    trainer.atomic_json(args.output_dir/'report.json', report)
    (args.output_dir/'selection_release.json').write_bytes(args.selection_release.read_bytes())
    (args.output_dir/'protocol.md').write_bytes(args.protocol.read_bytes())
    try:
        h, _ = trainer.load_helpers(args.repo)
        dataset, _, info = trainer.load_admitted_dataset(args, h)
        report['training_data'] = info
        for index, case in enumerate(schedule):
            _, _, tensors = selector.tensor_batch(h, dataset, case, index)
            arrays = {key: value.cpu().numpy() for key, value in tensors.items()}
            path = args.output_dir/(case['case']+'_batch.npz')
            with path.open('xb') as stream:
                h.np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            row = {**case, 'noise_seed': 0, 'noise_step': index, 'batch_file': path.name, 'batch_sha256': sha(path)}
            check_saved_batch(h, row, args.output_dir, index)
            report['cases'].append(row); trainer.atomic_json(args.output_dir/'report.json', report)
        trainer.reverify_goop_evidence_and_auxiliaries(args, h)
        h.data_loader.load_manifest_data(args.train_manifest, verify_hashes=True)
        require(sha(args.admission) == report['training_admission_sha256']
                and sha(args.selection_release) == report['selection_release_sha256']
                and sha(args.protocol) == report['protocol_sha256'] and sha(__file__) == report['source_sha256']
                and all(sha(Path(__file__).parent/name) == value for name, value in PINS.items())
                and all(sha(args.repo/name) == value for name, value in trainer.SOURCE_PINS.items()),
                'Preparation inputs or pinned sources changed')
        report.update(status='complete_saved_training_batches', elapsed_seconds=time.perf_counter()-started)
        trainer.atomic_json(args.output_dir/'report.json', report)
        return 0
    except BaseException as error:
        report.update(status='failed_batch_preparation', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter()-started, all_existing_outputs_retained=True)
        trainer.atomic_json(args.output_dir/'report.json', report)
        raise


def check_saved_batch(h, case, directory, index):
    require(case['case'] == CASES[index] and case['noise_seed'] == 0 and case['noise_step'] == index,
            'Prospective case/noise identity differs')
    path = Path(directory)/case['batch_file']
    require(path.name == CASES[index]+'_batch.npz' and path.resolve().parent == Path(directory).resolve()
            and sha(path) == case['batch_sha256'], 'Saved batch bytes or path differ')
    keys = {'position_sequence', 'particle_types', 'nparticles_per_example', 'next_positions', 'position_sequence_noise'}
    with h.np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == keys, 'Saved batch fields differ')
        values = {key: archive[key].copy() for key in keys}
    counts = values['nparticles_per_example']; n = int(counts.sum())
    require(counts.dtype == h.np.dtype('int64') and counts.tolist() == case['particle_counts']
            and len(counts) == 2 and all(int(x) > 0 for x in counts), 'Saved counts differ')
    for key, shape in (('position_sequence', (n, 6, 2)), ('position_sequence_noise', (n, 6, 2)), ('next_positions', (n, 2))):
        require(values[key].shape == shape and values[key].dtype == h.np.dtype('float32')
                and h.np.isfinite(values[key]).all(), 'Saved float batch shape/dtype/finiteness differs')
    require(values['particle_types'].shape == (n,) and values['particle_types'].dtype == h.np.dtype('int64')
            and h.np.all(values['particle_types'] == 7), 'Preserved Goop type7 vectors required')
    batch = {key: h.torch.from_numpy(value) for key, value in values.items()}
    expected_noise = h.host_noise(batch['position_sequence'].shape, batch['particle_types'], 0, index)
    require(h.torch.equal(expected_noise.view(h.torch.uint8), batch['position_sequence_noise'].view(h.torch.uint8)),
            'Saved host noise differs from prospective seed/case schedule')
    return batch, path


def validate_batch_report(report, trainer, schedule):
    require(report.get('schema') == BATCH_SCHEMA and report.get('status') == 'complete_saved_training_batches'
            and report.get('source_sha256') == sha(__file__) and report.get('helper_source_sha256') == PINS
            and report.get('train_manifest_sha256') == trainer.MANIFEST_SHA and report.get('metadata_sha256') == trainer.METADATA_SHA
            and report.get('training_admission_sha256') == ADMISSION_SHA
            and report.get('selection_uses_outcomes') is False and report.get('new_optimizer_updates') == 0
            and report.get('test_accessed') is False and report.get('schedule') == schedule
            and len(report.get('cases', [])) == 3, 'Exact prospective complete saved-batch report required')
    for row, expected in zip(report['cases'], schedule):
        require(all(row.get(key) == value for key, value in expected.items()), 'Saved batch source schedule differs')


def validate_execution_release(release, trainer, args, previous):
    require(release.get('schema') == VALIDATION_RELEASE and release.get('status') == 'admitted_for_bounded_validation'
            and release.get('issued_by') == 'root' and release.get('scientific_training_admitted') is False
            and release.get('batch_report_sha256') == sha(args.batch_report)
            and release.get('selection_release_sha256') == previous['selection_release_sha256']
            and release.get('training_admission_sha256') == previous['training_admission_sha256']
            and release.get('trainer_sha256') == TRAINER_SHA and release.get('validation_source_sha256') == sha(__file__)
            and release.get('protocol_sha256') == sha(args.protocol) == previous['protocol_sha256']
            and release.get('metadata_sha256') == trainer.METADATA_SHA and release.get('cuda_index') == args.cuda_index,
            'Exact root bounded-validation release required')
    checked = datetime.fromisoformat(release['process_identity_checked_utc'])
    require(checked.tzinfo is not None and -60 <= (datetime.now(timezone.utc)-checked).total_seconds() <= 300,
            'Fresh root process-identity check required')


def execute(args, mods):
    trainer, core = mods['train_goop_graph_support_cuda.py'], mods['validate_sand_graph_support_cuda.py']
    selector, compare = mods['validate_sand_cuda.py'], mods['validate_cuda_execution.py']
    require(sha(args.train_manifest) == trainer.MANIFEST_SHA and sha(args.metadata) == trainer.METADATA_SHA,
            'Exact Goop manifest/metadata required')
    previous = read(args.batch_report); schedule = expected_schedule(read(args.train_manifest), selector)
    validate_batch_report(previous, trainer, schedule)
    release = read(args.validation_release); validate_execution_release(release, trainer, args, previous)
    require(sha(args.batch_report.parent/'selection_release.json') == previous['selection_release_sha256'],
            'Saved prospective selection release differs')
    selection = read(args.batch_report.parent/'selection_release.json')
    require(selection.get('schema') == SELECTION_RELEASE and selection.get('status') == 'admitted_for_batch_preparation'
            and selection.get('issued_by') == 'root' and selection.get('scientific_training_admitted') is False
            and selection.get('schedule') == schedule and selection.get('training_admission_sha256') == ADMISSION_SHA
            and selection.get('train_manifest_sha256') == trainer.MANIFEST_SHA
            and selection.get('trainer_sha256') == TRAINER_SHA and selection.get('preparation_source_sha256') == sha(__file__)
            and selection.get('protocol_sha256') == sha(args.protocol), 'Saved root selection release contract differs')
    require(shared_numeric_check(), 'Pinned Goop numerical/run-loop AST differs from Sand origin')
    args.output_dir.mkdir(mode=0o700, exist_ok=False); started = time.perf_counter()
    report = {**describe(), 'status': 'in_progress', 'source_sha256': sha(__file__), 'cases': [], 'replay': {},
              'shared_trainer_functions_exact': True, 'all_inputs_reverified': False,
              'batch_report_sha256': sha(args.batch_report), 'validation_release_sha256': sha(args.validation_release),
              'selection_release_sha256': previous['selection_release_sha256'],
              'train_manifest_sha256': trainer.MANIFEST_SHA, 'training_admission_sha256': previous['training_admission_sha256'],
              'protocol_sha256': sha(args.protocol)}
    def save():
        trainer.atomic_json(args.output_dir/'report.json', report)
    save()
    try:
        h, support = trainer.load_helpers(args.repo)
        device, runtime = trainer.configure_cuda(h, args)
        require(runtime['uuid'].removeprefix('GPU-').lower() == release['gpu_uuid'].removeprefix('GPU-').lower(), 'Released GPU UUID differs')
        report['runtime'] = runtime; metadata = read(args.metadata)
        h.torch.manual_seed(0)
        with h.torch.cuda.device(device):
            h.torch.cuda.manual_seed(0)
        initial_model = selector.make_model(h, metadata, 'faithful', device)
        initial = h.cpu_tree(initial_model.state_dict()); initial_rng = trainer.capture_rng(h.torch, device)
        report['initial_state'] = core.save_evidence(h.torch, args.output_dir/'initial_state_seed0.pt', initial)
        del initial_model
        make_model = lambda state: selector.make_model(h, metadata, 'faithful', device, state)
        inputs = {args.batch_report: report['batch_report_sha256'], args.metadata: trainer.METADATA_SHA,
                  args.train_manifest: trainer.MANIFEST_SHA, args.protocol: report['protocol_sha256'],
                  args.batch_report.parent/'selection_release.json': previous['selection_release_sha256'],
                  args.validation_release: report['validation_release_sha256'], Path(__file__): report['source_sha256'],
                  **{Path(__file__).parent/name: value for name, value in PINS.items()},
                  **{args.repo/name: value for name, value in trainer.SOURCE_PINS.items()}}
        batches = []
        for index, case in enumerate(previous['cases']):
            batch, path = check_saved_batch(h, case, args.batch_report.parent, index)
            inputs[path] = case['batch_sha256']; batches.append((batch, case))
            row = {'case': case['case'], 'batch_file': path.name, 'batch_sha256': sha(path),
                   'graph_schedule_step': index, 'particle_counts': case['particle_counts'], 'branches': {}}
            report['cases'].append(row); save(); results = {}
            for arm in BRANCHES:
                model = make_model(initial); opt = core.optimizer(h, model)
                trainer.restore_rng(h.torch, initial_rng, device)
                result, check, _ = core.advance(h, trainer, support, compare, model, opt, batch,
                    batch['position_sequence_noise'], device, 0, arm, graph_step=index)
                results[arm] = result
                check['evidence'] = core.save_evidence(h.torch, args.output_dir/f"{case['case']}_{arm}.pt", result)
                row['branches'][arm] = check; del model, opt; save()
            row['old_native_vs_base'] = core.branch_comparison(h, trainer, compare, results['old_native'], results['base'])
            row['mix_target_exact'] = compare.diff(h.torch, results['base']['before']['target'], results['mix']['before']['target'], 'exact')
            del results; save()
        for arm in ('base', 'mix'):
            report['replay'][arm] = core.replay(h, trainer, support, compare, make_model, initial, initial_rng,
                batches[0][0], batches[0][1], device, arm, args.output_dir); save()
        require(all(sha(path) == value for path, value in inputs.items()), 'Source/saved input changed during validation')
        passed = core.completion_gate(report)
        report.update(status='implementation_passed' if passed else 'implementation_failed',
                      all_inputs_reverified=True, elapsed_seconds=time.perf_counter()-started)
        save(); return 0 if passed else 2
    except BaseException as error:
        if hasattr(error, 'graph_validation_evidence'):
            report['failed_update_evidence'] = core.save_evidence(h.torch, args.output_dir/'failed_update_evidence.pt', error.graph_validation_evidence)
        report.update(status='implementation_error', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter()-started, all_existing_outputs_retained=True)
        save(); raise


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--prepare-batches', action='store_true'); mode.add_argument('--execute', action='store_true')
    for name in ('repo', 'train-manifest', 'admission', 'structural-report', 'acquisition-report', 'context-semantics',
                 'auxiliary-report', 'selection-release', 'batch-report', 'metadata', 'validation-release', 'protocol', 'output-dir'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--cuda-index', type=int, default=0); p.add_argument('--threads', type=int, default=2)
    a = p.parse_args(argv)
    required = ('repo', 'train_manifest', 'protocol', 'output_dir')
    if a.prepare_batches:
        required += ('admission', 'structural_report', 'acquisition_report', 'context_semantics', 'auxiliary_report', 'selection_release')
    if a.execute:
        required += ('batch_report', 'metadata', 'validation_release')
    if (a.prepare_batches or a.execute) and any(getattr(a, key) is None for key in required):
        p.error('Every mode-specific source/release/input/output path is required')
    if a.cuda_index < 0 or a.threads != 2:
        p.error('Nonnegative CUDA index and exactly2 threads required')
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.prepare_batches and not a.execute:
        print(json.dumps(describe(), indent=2)); return 0
    mods = modules()
    return prepare_batches(a, mods) if a.prepare_batches else execute(a, mods)


if __name__ == '__main__':
    raise SystemExit(main())
