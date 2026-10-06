#!/usr/bin/env python3
"""CPU-only graph-stage profile; no GNS/model import, CUDA call or optimizer.

Uses CPU Torch solely for the original private RNG and CPU graph tensor API.
Description only by default. Root owns actual data execution and timeout.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
BENCHMARK_SHA = '8cfaa4fcf16afaa459b805445b0a6c25282919dd23776872fe353daa06043722'
GRAPH_SHA = '7d43fe7d06ac450b6b9031181902cfc6d3d9754af3a26e17e96fd46c4a1bf64f'
REFERENCE_SHA = '500a333a397f706f1e0ff9d482cb7389279d0ab5fdb6bdca0c8f612d14e26367'
SCHEMA = 'goop3d_first_record_cpu_graph_stage_profile_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load_pinned_helpers(directory):
    benchmark_path = directory / 'benchmark_goop3d_first_record_v2.py'
    graph_path = directory / 'goop3d_graph_support.py'
    require(sha(benchmark_path) == BENCHMARK_SHA and sha(graph_path) == GRAPH_SHA, 'Pinned helper bytes differ')
    spec = importlib.util.spec_from_file_location('_goop3d_cpu_profile_input', benchmark_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)  # Standard-library imports only; no model loader called.
    import numpy as np
    from scipy.spatial import cKDTree
    import torch
    names = {'Candidates', 'require', 'state_hash', 'canonical_pairs', 'pair_set', 'check_candidate_bound',
             'strict_pairs', 'ordered_edges', 'graph_rng', 'append_optional_edges', 'host_noise'}
    tree = ast.parse(graph_path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    require({node.name for node in selected} == names, 'Pinned pure/CPU graph functions missing')
    namespace = {'np': np, 'torch': torch, 'cKDTree': cKDTree, 'math': math, 'hashlib': hashlib,
                 'dataclass': dataclass, '__name__': __name__, 'RADIUS_FACTOR': 1.267,
                 'MAX_PAIRS': 2000000, 'MAX_BATCH_EDGES': 5000000, 'NOISE': 6.7e-4, 'KINEMATIC': 3}
    # Execute only named unchanged function/class definitions, excluding the
    # graph module's imports and all model-forward functions.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(graph_path), 'exec'), namespace)
    return bench, SimpleNamespace(**namespace)


def native_graph_cpu(points, counts, np, cKDTree, radius=.025):
    """Literal CPU algorithm of pinned LearnedSimulator native connectivity.

    No optimized replacement: same query_ball_point, float32 norm, strict
    filter, distance/source-ID sort and128receiver cap including self candidates.
    """
    sources, targets, offset = [], [], 0
    for count in counts:
        local = points[offset:offset + count]
        tree = cKDTree(local)
        for target, neighbors in enumerate(tree.query_ball_point(local, radius)):
            neighbors = np.asarray(neighbors, dtype=np.int64)
            distances = np.linalg.norm(local[neighbors] - local[target], axis=1)
            keep = distances < radius
            neighbors, distances = neighbors[keep], distances[keep]
            order = np.lexsort((neighbors, distances))[:128]
            sources.extend((neighbors[order] + offset).tolist())
            targets.extend([target + offset] * len(order))
        offset += count
    return np.asarray([sources, targets], dtype=np.int64)


def graph_stages(batch, noise, case, graph):
    # CPU tensor construction matches the v2 first-record noisy state exactly;
    # completed v2 hash comparisons below are mandatory before timing is useful.
    require(all(value.device.type == 'cpu' for value in (*batch, noise)), 'CPU tensors required')
    noisy = batch[0] + noise
    points = noisy[:, -1].numpy()
    counts = batch[2].tolist()
    row = {'stage_seconds': {}}
    started = time.perf_counter()
    offset = 0
    for count in counts:
        graph.check_candidate_bound(points[offset:offset + count], .025)
        offset += count
    row['stage_seconds']['candidate_precheck'] = time.perf_counter() - started
    started = time.perf_counter()
    native = native_graph_cpu(points, counts, graph.np, graph.cKDTree)
    native_tensor = graph.torch.from_numpy(native)
    row['stage_seconds']['native_connectivity_cpu'] = time.perf_counter() - started
    started = time.perf_counter()
    edges, ledger = graph.append_optional_edges(noisy, batch[2], native_tensor, .025, 0, 0, case)
    row['stage_seconds']['optional_graph_and_ledger_cpu'] = time.perf_counter() - started
    row.update(graph_ledger=ledger, native_batch_edge_sha256=graph.state_hash(native),
               final_batch_edge_sha256=graph.state_hash(edges.numpy()),
               final_directed_edges=int(edges.shape[1]), summed_graph_stage_seconds=sum(row['stage_seconds'].values()))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--execute', action='store_true')
    for name in ('inspection-dir', 'metadata', 'official-reading-utils', 'reference-report', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'schema': SCHEMA, 'execution': False, 'cuda_calls': False, 'model_or_optimizer': False,
                          'cases': ['base', 'expanded25'], 'warmup_repetitions': 1, 'measured_repetitions': 3}))
        return 0
    require(all(getattr(args, name) is not None for name in
                ('inspection_dir', 'metadata', 'official_reading_utils', 'reference_report', 'output')), 'Explicit pinned inputs/fresh output required')
    require(not args.output.exists(), 'Preserve existing profile output')
    directory = Path(__file__).resolve().parent
    bench, graph = load_pinned_helpers(directory)
    report = {'schema': SCHEMA, 'status': 'running', 'source_sha256': sha(__file__), 'cuda_calls': False,
              'model_or_optimizer': False, 'scientific_training_admitted': False, 'test_accessed': False, 'cases': []}
    bench.write(args.output, report)
    started = time.perf_counter()
    try:
        require(sha(args.reference_report) == REFERENCE_SHA, 'Completed v2 reference report differs')
        reference = json.loads(args.reference_report.read_text())
        require(reference['status'] == 'complete_fixed_state_capacity_only'
                and reference['script_sha256'] == BENCHMARK_SHA, 'Completed v2 reference required')
        graph.torch.set_num_threads(2)
        batch, noise, metadata, info, inputs = bench.load_inputs(args, graph)
        for key in ('history_sha256', 'labels_sha256', 'noise_sha256'):
            require(info[key] == reference['input'][key], 'Same-state/noise hash mismatch: ' + key)
        report.update(input=info, input_files_sha256=inputs, reference_report_sha256=REFERENCE_SHA,
                      runtime={'python': platform.python_version(), 'platform': platform.platform(),
                               'torch': str(graph.torch.__version__), 'numpy': graph.np.__version__, 'cpu_threads': 2})
        for case in ('base', 'expanded25'):
            expected = next(row for row in reference['cases'] if row['case'] == case)['updates'][0]['graph_ledger']
            case_report = {'case': case, 'repetitions': []}
            report['cases'].append(case_report)
            for repeat in range(4):
                row = graph_stages(batch, noise, case, graph)
                require(row['graph_ledger'] == expected, 'CPU graph does not match the completed CUDA benchmark graph ledger')
                row.update(repetition=repeat, warmup=repeat == 0, exact_reference_graph_ledger=True)
                case_report['repetitions'].append(row)
                bench.write(args.output, report)
            case_report['mean_measured_stage_seconds'] = {name: statistics.mean(row['stage_seconds'][name]
                for row in case_report['repetitions'][1:]) for name in case_report['repetitions'][0]['stage_seconds']}
            case_report['mean_measured_graph_seconds'] = statistics.mean(row['summed_graph_stage_seconds']
                                                                       for row in case_report['repetitions'][1:])
        require(all(bench.sha(path) == digest for path, digest in inputs.items())
                and sha(args.reference_report) == REFERENCE_SHA and sha(__file__) == report['source_sha256']
                and sha(directory / 'benchmark_goop3d_first_record_v2.py') == BENCHMARK_SHA
                and sha(directory / 'goop3d_graph_support.py') == GRAPH_SHA, 'Profile inputs/source changed')
        report.update(status='complete_cpu_graph_stage_profile_only',
                      limitations=['CPU host/load can differ from original CUDA benchmark; do not directly subtract timings as a proven speedup.',
                                   'No GPU graph-transfer, GNN, backward or optimizer timing here; no numerical optimizer/model result.',
                                   'One fixed training state, no population capacity or efficacy claim.'])
        return 0
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        bench.write(args.output, report)


if __name__ == '__main__':
    raise SystemExit(main())
