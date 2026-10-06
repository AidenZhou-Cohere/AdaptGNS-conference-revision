#!/usr/bin/env python3
"""Opt-in bounded Goop-3D first-record CUDA timing; never an efficacy study.

Two fresh paired faithful models, base versus forced full exposure to 25% of
annulus pairs. One warmup plus three measured updates each: eight updates total.
Root owns process inventory and a 600-second outer timeout. No test, acquisition,
resume, checkpoint promotion, CPU fallback or whole-cohort training forecast.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = 'goop3d_first_training_record_capacity_v2'
CORE = {
    'device_utils.py': '324db853f98a85a6a2f83755cb1d67e18edac2f503e5dc33d6bd39352749b326',
    'graph_network.py': 'af91d949063441ba876fbfbaefceee65f7f16947fe98053720a1ccb15feffe91',
    'learned_simulator.py': '216c73a236da6e2441828c4942068cc19618d339859d6492480f07ae2429eadf',
    'losses.py': '94ba3f2eb6f103527801f251919675e77e242fff262728f03f0e87cd4e2ea0e5',
    'model_io.py': '06599b723814f0be507a3f4210abe10e21f8ada7deeda1acaced07351d82de2e',
}
GRAPH_SHA = '7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f'
INSPECTION_SHA = 'c003d147b93eb0eb2cebafe7c9a79ac4179d7224fd41d139e1719f5ceefedd78'
METADATA_SHA = '727cec55bc529c142596954773284e897aaed257e907ab6439d5c367853e4a55'
OFFICIAL_READER_SHA = '5868022f76eaf15b2626125bdaa3c973ccaf2dfc0b0ec50d9e05d0c3d973e149'
CASES = ('base', 'expanded25')
WARMUPS, MEASURED, SEED, TARGET = 1, 3, 0, 6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def description():
    return {'schema': SCHEMA, 'status': 'description_only', 'scientific_training_admitted': False,
            'test_accessed': False, 'cases': list(CASES), 'warmup_updates_per_case': WARMUPS,
            'measured_updates_per_case': MEASURED, 'total_optimizer_updates': 8,
            'data': 'two copies of first training record target6; same fixed host noise in both cases/all updates',
            'architecture': '3D;37 node features;4 edge features;128 width;10 blocks;2-layer MLPs',
            'radius': .025, 'noise_std': 6.7e-4, 'likelihood_log_coefficient': 1.5, 'vector_risk': '3q',
            'scope': 'synchronized fixed-state update time and peak CUDA memory, no efficacy/full-cohort forecast',
            'outer_timeout_seconds': 600, 'max_candidate_pairs_per_example': 2000000,
            'max_directed_edges_per_batch': 5000000}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    for name in ('repo', 'inspection-dir', 'metadata', 'official-reading-utils', 'output-dir'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--cuda-index', type=int, default=0)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    require(args.cuda_index >= 0 and 1 <= args.threads <= 16, 'Invalid device index or thread count')
    if args.execute:
        require(all(getattr(args, name) is not None for name in
                    ('repo', 'inspection_dir', 'metadata', 'official_reading_utils', 'output_dir', 'gpu_uuid')),
                'Explicit repository, first-record input, metadata, official parser, fresh output and GPU UUID required')
    return args


def verify_sources(repo, graph_path):
    root = Path(repo).resolve() / 'adaptive-gns/gns'
    actual = {name: sha(root / name) for name in CORE}
    require(actual == CORE and sha(graph_path) == GRAPH_SHA, 'Frozen core or separate 3D adapter hash differs')
    return root


def load_helpers(repo, graph_path):
    root = verify_sources(repo, graph_path)
    for filename in CORE:
        name = 'gns.' + filename[:-3]
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve() == root / filename, 'Unrelated cached core module')
    sys.path.insert(0, str(root.parent))
    model_io = importlib.import_module('gns.model_io')
    losses = importlib.import_module('gns.losses')
    spec = importlib.util.spec_from_file_location('_goop3d_benchmark_graph', graph_path)
    graph = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = graph
    spec.loader.exec_module(graph)
    for filename in CORE:
        require(Path(sys.modules['gns.' + filename[:-3]].__file__).resolve() == root / filename,
                'Imported core module does not match the pinned path')
    return graph, model_io, losses


def load_inputs(args, graph):
    report_path = args.inspection_dir / 'report.json'
    require(sha(report_path) == INSPECTION_SHA, 'Only the reviewed first training record is accepted')
    inspection = json.loads(report_path.read_text())
    require(inspection['status'] == 'first_training_record_verified' and inspection['length_crc_verified']
            and inspection['payload_crc_verified'] and inspection['dimension'] == 3
            and inspection['particles'] == 9271 and inspection['stored_frames'] == 301,
            'First-record inspection contract differs')
    require(sha(args.metadata) == METADATA_SHA and sha(args.official_reading_utils) == OFFICIAL_READER_SHA,
            'Metadata or pinned official parser source differs')
    metadata = json.loads(args.metadata.read_text())
    require(metadata['dim'] == 3 and metadata['sequence_length'] == 300
            and metadata['default_connectivity_radius'] == .025
            and 'context_mean' not in metadata and 'context_std' not in metadata,
            'Goop-3D geometry or official context-omission contract differs')
    paths = {str(report_path): INSPECTION_SHA, str(args.metadata): METADATA_SHA,
             str(args.official_reading_utils): OFFICIAL_READER_SHA}
    arrays = {}
    for field, basename in [('positions', 'position_000000.npy'), ('particle_types', 'type_000000.npy'),
                            ('step_context', 'step_context_000000.npy')]:
        row = inspection['record'][field]
        require(row['path'] == 'train/' + basename, 'Unexpected first-record array path')
        # The bounded inspector passes OUT/numeric as decode_record staging.
        # Its manifest-style train/ descriptor is not the on-disk directory.
        path = args.inspection_dir / 'numeric' / basename
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['size_bytes']
                and sha(path) == row['sha256'], 'First-record numeric-array identity differs')
        value = graph.np.load(path, mmap_mode='r', allow_pickle=False)
        require(value.dtype.str == row['dtype'] and list(value.shape) == row['shape'], 'Array shape/dtype differs')
        arrays[field] = value
        paths[str(path)] = row['sha256']
    positions, types = arrays['positions'], arrays['particle_types']
    require(positions.shape == (301, 9271, 3) and graph.np.isfinite(positions).all(), 'Finite full first record required')
    require(types.shape == (9271,) and graph.np.all((types >= 0) & (types <= 8))
            and graph.np.any(types != 3), 'GNS type0..8 and at least one dynamic particle required')
    history = graph.np.ascontiguousarray(positions[:TARGET].transpose(1, 0, 2))
    next_positions = graph.np.ascontiguousarray(positions[TARGET])
    torch, np = graph.torch, graph.np
    batch = (torch.from_numpy(np.concatenate((history, history))),
             torch.from_numpy(np.concatenate((types, types))),
             torch.tensor([9271, 9271], dtype=torch.long),
             torch.from_numpy(np.concatenate((next_positions, next_positions))))
    noise = graph.host_noise(batch[0].shape, batch[1], SEED, 0)
    aux = arrays['step_context']
    info = {'trajectory': 'train:000000', 'target_frames': [TARGET, TARGET], 'batch_particles': [9271, 9271],
            'observed_type_ids': [int(v) for v in np.unique(types)],
            'history_sha256': graph.state_hash(batch[0].numpy()), 'labels_sha256': graph.state_hash(batch[3].numpy()),
            'noise_sha256': graph.state_hash(noise.numpy()),
            'auxiliary': {'shape': list(aux.shape), 'nan_count': int(np.isnan(aux).sum()),
                          'inf_count': int(np.isinf(aux).sum()), 'finite_count': int(np.isfinite(aux).sum()),
                          'handling': 'preserved/excluded: official parser returns no context when context_mean is absent'},
            'whole_dataset_admission': False, 'whole_source_checksum_claim': False}
    return batch, noise, metadata, info, paths


def normalized_uuid(value):
    return str(value).lower().removeprefix('gpu-')


def configure_cuda(torch, args):
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Start with CUBLAS_WORKSPACE_CONFIG=:4096:8')
    require(str(torch.__version__) == '2.13.0+cu129' and torch.version.cuda == '12.9', 'Validated CUDA stack required')
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Leave CUDA_VISIBLE_DEVICES unset; bind physical UUID explicitly')
    require(torch.cuda.is_available() and args.cuda_index < torch.cuda.device_count(), 'Requested CUDA unavailable; no fallback')
    device = torch.device('cuda', args.cuda_index)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    props = torch.cuda.get_device_properties(device)
    require('GB200' in props.name and normalized_uuid(props.uuid) == normalized_uuid(args.gpu_uuid),
            'Current device differs from root-selected GB200 UUID')
    return device, {'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': str(device),
                    'device_name': props.name, 'gpu_uuid': str(props.uuid), 'total_memory': props.total_memory,
                    'threads': args.threads, 'tf32': False, 'amp': False, 'compile': False,
                    'deterministic_algorithms': True, 'python': platform.python_version()}


def model_hash(model, graph):
    values = [(name, graph.state_hash(value.detach().cpu().numpy())) for name, value in model.state_dict().items()]
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def step_once(model, optimizer, batch, noise, device, case, graph, losses, context):
    torch = graph.torch
    optimizer.zero_grad(set_to_none=True)
    prediction, head, target, ledger = graph.forward_batch(model, batch, noise, device, SEED, 0, case)
    context['graph_ledger'] = ledger
    require(prediction.shape == target.shape == (len(batch[0]), 3)
            and head.shape == (len(batch[0]),), 'D3 forward shape differs')
    if not all(bool(torch.isfinite(value).all()) for value in (prediction, head, target)) or not bool((head > 0).all()):
        raise FloatingPointError('Nonfinite prediction/target/variance or nonpositive variance')
    mask = (batch[1] != 3).to(device)
    loss = losses.acceleration_loss(prediction, target, mask, pred_variance=head, loss_type='faithful', variance_floor=1e-6)
    context['loss_text'] = repr(float(loss.detach().cpu()))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite faithful D3 loss; no optimizer update')
    loss.backward()
    if any(value.grad is None or not bool(torch.isfinite(value.grad).all()) for value in model.parameters()):
        raise FloatingPointError('Missing/nonfinite gradient; no optimizer update')
    context['optimizer_step_called'] = True
    optimizer.step()
    context['optimizer_step_returned'] = True
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise FloatingPointError('Nonfinite parameter after update')
    diagnostics = graph.normalized_statistics(prediction, head, target, mask)
    return float(loss.detach().cpu()), ledger, diagnostics


def interrupted(signum, _frame):
    raise InterruptedError('Root timeout or interruption signal ' + str(signum))


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(description()))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / 'report.json'
    start = time.perf_counter()
    graph_path = Path(__file__).with_name('goop3d_graph_support.py')
    report = {**description(), 'status': 'running', 'pid': os.getpid(), 'hostname': socket.gethostname(),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [], 'attempted_update': None,
              'script_sha256': sha(__file__), 'graph_adapter_sha256': GRAPH_SHA, 'core_sha256': CORE}
    write(report_path, report)
    old_handlers = {value: signal.signal(value, interrupted) for value in (signal.SIGINT, signal.SIGTERM)}
    try:
        graph, model_io, losses = load_helpers(args.repo, graph_path)
        batch, noise, metadata, info, input_paths = load_inputs(args, graph)
        report['input'] = info
        report['input_files_sha256'] = input_paths
        torch = graph.torch
        device, runtime = configure_cuda(torch, args)
        report['runtime'] = {**runtime, 'numpy': graph.np.__version__, 'platform': platform.platform()}
        first_initial_hash = None
        for case in CASES:
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            model = model_io.build_simulator(metadata, graph.NOISE, graph.NOISE, device,
                connectivity_radius=.025, nmessage_passing_steps=10, uncertainty_parameterization='variance',
                variance_floor=1e-6, detach_variance_features=True, radius_backend='scipy_host').to(device).train()
            require(model._checkpoint_config['nnode_in'] == 37 and model._checkpoint_config['nedge_in'] == 4,
                    'D3 model feature dimensions differ')
            initial_hash = model_hash(model, graph)
            if first_initial_hash is None:
                first_initial_hash = initial_hash
            require(initial_hash == first_initial_hash, 'Fresh cases have different initialization')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                         weight_decay=0., foreach=False, fused=False)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            free, total = torch.cuda.mem_get_info(device)
            case_report = {'case': case, 'status': 'running', 'initial_state_sha256': initial_hash,
                           'free_device_bytes_before_updates': free, 'device_total_bytes': total,
                           'updates': [], 'completed_optimizer_updates': 0}
            report['cases'].append(case_report)
            write(report_path, report)
            for index in range(WARMUPS + MEASURED):
                context = {'case': case, 'case_update_index': index, 'warmup': index < WARMUPS}
                report['attempted_update'] = context
                write(report_path, report)
                torch.cuda.synchronize(device)
                update_start = time.perf_counter()
                loss, ledger, diagnostics = step_once(model, optimizer, batch, noise, device, case, graph, losses, context)
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - update_start
                row = {**context, 'loss': loss, 'synchronized_update_seconds': seconds,
                       'peak_allocated_bytes_since_case_start': torch.cuda.max_memory_allocated(device),
                       'peak_reserved_bytes_since_case_start': torch.cuda.max_memory_reserved(device),
                       'diagnostics_before_update': diagnostics}
                case_report['updates'].append(row)
                case_report['completed_optimizer_updates'] += 1
                write(report_path, report)
            values = [row['synchronized_update_seconds'] for row in case_report['updates'][WARMUPS:]]
            case_report.update(status='complete', measured_mean_seconds=statistics.mean(values),
                               measured_median_seconds=statistics.median(values), measured_min_seconds=min(values),
                               measured_max_seconds=max(values), final_state_sha256=model_hash(model, graph))
            write(report_path, report)
            del model, optimizer
        for path, expected in input_paths.items():
            require(sha(path) == expected, 'First-record input changed during benchmark')
        verify_sources(args.repo, graph_path)
        require(sha(__file__) == report['script_sha256'], 'Benchmark source changed during execution')
        require(sum(row['completed_optimizer_updates'] for row in report['cases']) == 8, 'Bounded update count differs')
        report.update(status='complete_fixed_state_capacity_only', attempted_update=None,
                      total_completed_optimizer_updates=8,
                      timed_scope='fixed precreated host-batch/noise transfer; native and optional graphs; forward; faithful loss; backward; Adam; finite checks; D3 diagnostics; CUDA sync',
                      excluded_from_update_timing='imports, source/data verification, batch/noise creation, model initialization, report IO and final hashes',
                      limitations=['One fixed training state and duplicated batch; no full-cohort distribution or runtime bound.',
                                   'Forced expanded25 exposes both examples, not the probabilistic mix training distribution.',
                                   'Three measured updates per case; no efficacy, calibration, rollout or CPU/CUDA equivalence claim.',
                                   'No checkpoint is published; benchmark states cannot initialize a scientific run.'])
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if report['cases'] and report['cases'][-1]['status'] == 'running':
            report['cases'][-1]['status'] = 'failed'
        raise
    finally:
        for value, handler in old_handlers.items():
            signal.signal(value, handler)
        report['elapsed_seconds'] = time.perf_counter() - start
        report['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(report_path, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
