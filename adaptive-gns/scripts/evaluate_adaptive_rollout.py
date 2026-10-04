#!/usr/bin/env python3
"""
Week 3: Evaluate adaptive rollout MSE at every step.

Each rollout step performs one neural pass. The first uses the base graph;
later steps expand it using the previous step's uncertainty ranking.

Results are saved in the same .npz / .json format as evaluate_rollout_mse.py so
all curves can be plotted together.
"""
import os
import sys
import argparse
import json
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns import learned_simulator, data_loader, reading_utils
from gns.model_io import load_for_evaluation
from gns.device_utils import add_device_argument, resolve_device

INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3
EVAL_STEPS = [1, 10, 50, 200, 500, 1000]


def adaptive_rollout_mse(simulator, positions, particle_type, material_property,
                         n_particles_per_example, nsteps, device,
                         sigma_percentile, radius_factor):
    """Single-pass lagged-sigma adaptive rollout; returns per-step MSE and edge counts.

    Step 0 uses the base graph (no prior sigma).  From step 1 onward the graph
    topology is built from sigma of the *previous* step, so each step requires
    exactly ONE GNN forward pass. Graph construction adds separate overhead.

    Returns:
      mse_per_step: np.ndarray (nsteps,)
      edge_counts: np.ndarray (nsteps,) — number of edges used at each step
    """
    initial_positions = positions[:, :INPUT_SEQUENCE_LENGTH]
    if nsteps < 1 or nsteps > positions.shape[1] - INPUT_SEQUENCE_LENGTH:
        raise ValueError("nsteps must be within the available forecast horizon")
    ground_truth = positions[:, INPUT_SEQUENCE_LENGTH:INPUT_SEQUENCE_LENGTH + nsteps]
    current = initial_positions

    kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).bool()
    non_kinematic = ~kinematic_mask
    if not bool(non_kinematic.any()):
        raise ValueError("MSE requires at least one dynamic particle")

    predictions = []
    edge_counts = []

    # Step 0: base graph — obtains sigma_0 which drives step 1's graph
    node_features, edge_index, edge_features = simulator._encoder_preprocessor(
        current,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_type,
        material_property=material_property,
    )
    edge_counts.append(edge_index.shape[1])
    pred_acc, sigma_prev = simulator._encode_process_decode(node_features, edge_index, edge_features)
    next_pos = simulator._decoder_postprocessor(pred_acc, current)

    gt_step = ground_truth[:, 0]
    next_pos = torch.where(
        kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
        gt_step, next_pos,
    )
    predictions.append(next_pos)
    current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    for step in range(1, nsteps):
        # Build adaptive graph using lagged sigma
        high_sigma_mask = simulator._select_high_uncertainty(
            sigma_prev, [n_particles_per_example], sigma_percentile)
        most_recent_position = current[:, -1]
        r_large = simulator._connectivity_radius * radius_factor
        senders, receivers = simulator._build_adaptive_edge_index(
            most_recent_position, [n_particles_per_example], high_sigma_mask, r_large)
        edge_counts.append(senders.shape[0])

        next_pos, sigma_prev = simulator._forward_with_edge_index(
            current,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_type,
            senders=senders,
            receivers=receivers,
            material_property=material_property,
        )
        gt_step = ground_truth[:, step]
        next_pos = torch.where(
            kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
            gt_step, next_pos,
        )
        predictions.append(next_pos)
        current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    predictions = torch.stack(predictions)          # (nsteps, nparticles, dim)
    gt = ground_truth.permute(1, 0, 2)              # (nsteps, nparticles, dim)

    se = (predictions - gt) ** 2
    se_masked = se * non_kinematic[None, :, None].float()
    n_non_kinematic = non_kinematic.sum().item()
    mse_per_step = se_masked.sum(dim=(1, 2)) / (n_non_kinematic * predictions.shape[-1])

    return mse_per_step.cpu().numpy(), np.array(edge_counts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate single-pass lagged-uncertainty rollout MSE')
    add_device_argument(parser)
    parser.add_argument('--data_path', required=True,
                        help='Dataset directory (contains test.npz, metadata.json)')
    parser.add_argument('--model_path', required=True,
                        help='Adaptive-GNS model directory')
    parser.add_argument('--model_file', default='latest',
                        help='Checkpoint filename or "latest"')
    parser.add_argument('--sigma_percentile', type=float, default=80.0,
                        help='Particles above this sigma percentile get expanded edges (default: 80)')
    parser.add_argument('--radius_factor', type=float, default=1.267,
                        help='Multiplier for expanded connectivity radius (default: 1.267)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Limit evaluation to N test trajectories')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path stem (default: <model_path>/adaptive_rollout_mse_<model>)')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='valid',
                        help='Use valid for selection; test only after settings are frozen.')
    parser.add_argument('--normalization_noise_std', type=float, default=None,
                        help='Legacy checkpoint training normalization noise (not input noise).')
    parser.add_argument('--nmessage_passing_steps', type=int, default=None,
                        help='Legacy checkpoint depth, including zero for an MLP.')
    parser.add_argument('--radius_backend', choices=['auto', 'pyg', 'scipy', 'scipy_host'], default=None,
                        help='Explicit runtime graph backend override; scipy_host builds on CPU and transfers edges')
    args = parser.parse_args()
    if args.max_trajectories is not None and args.max_trajectories <= 0:
        parser.error('--max_trajectories must be positive')

    data_path = args.data_path.rstrip('/') + '/'
    model_path = args.model_path.rstrip('/') + '/'

    if args.model_file == 'latest':
        candidates = glob.glob(os.path.join(model_path, 'model-*.pt'))
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints in {model_path}")
        model_file = max(candidates, key=lambda p: int(p.split('-')[-1].split('.')[0]))
        model_file = os.path.basename(model_file)
    else:
        model_file = args.model_file

    output_stem = args.output
    if output_stem is None:
        model_basename = model_file.replace('.pt', '')
        output_stem = os.path.join(
            model_path,
            f'adaptive_rollout_mse_{model_basename}_{args.split}_p{args.sigma_percentile:g}_r{args.radius_factor}')
    # Strip any extension that might have been passed
    for ext in ('.npz', '.json'):
        if output_stem.endswith(ext):
            output_stem = output_stem[:-len(ext)]

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Model:  {model_file}")
    print(f"Sigma percentile: {args.sigma_percentile}  |  Radius factor: {args.radius_factor}")

    metadata = reading_utils.read_metadata(data_path, 'rollout')
    simulator, provenance = load_for_evaluation(
        os.path.join(model_path, model_file), metadata, device,
        normalization_noise_std=args.normalization_noise_std,
        radius_backend=args.radius_backend,
        nmessage_passing_steps=args.nmessage_passing_steps)

    ds = data_loader.get_data_loader_by_trajectories(path=data_path + args.split + '.npz')
    has_material = len(ds.dataset._data[0]) == 3

    mse_at_steps = {s: [] for s in EVAL_STEPS}
    mse_all_trajectories = []
    edge_counts_all = []

    n_eval = 0
    with torch.no_grad():
        for i, features in enumerate(ds):
            if args.max_trajectories and i >= args.max_trajectories:
                break
            positions = features[0].to(device)
            particle_type = features[1].to(device)
            if has_material:
                material_property = features[2].to(device)
                n_particles = int(features[3])
            else:
                material_property = None
                n_particles = int(features[2])

            nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH

            mse_per_step, edge_counts = adaptive_rollout_mse(
                simulator, positions, particle_type, material_property,
                n_particles, nsteps, device,
                args.sigma_percentile, args.radius_factor,
            )

            mse_all_trajectories.append(mse_per_step)
            edge_counts_all.append(edge_counts)

            for s in EVAL_STEPS:
                idx = s - 1
                if idx < len(mse_per_step):
                    mse_at_steps[s].append(float(mse_per_step[idx]))

            n_eval += 1
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1} trajectories...")

    if not mse_all_trajectories:
        raise ValueError("No trajectories evaluated")
    max_steps = max(len(m) for m in mse_all_trajectories)
    mse_stacked = np.full((n_eval, max_steps), np.nan)
    for i, m in enumerate(mse_all_trajectories):
        mse_stacked[i, :len(m)] = m

    steps = np.arange(1, max_steps + 1, dtype=np.int32)
    mse_mean = np.nanmean(mse_stacked, axis=0)
    mse_std = np.nanstd(mse_stacked, axis=0)
    n_at_step = np.sum(~np.isnan(mse_stacked), axis=0)

    # Edge count stats across all trajectories and steps
    edge_counts_stacked = np.full((n_eval, max_steps), np.nan)
    for i, ec in enumerate(edge_counts_all):
        edge_counts_stacked[i, :len(ec)] = ec
    mean_edges_per_step = float(np.nanmean(edge_counts_stacked))
    mean_edges_by_step = np.nanmean(edge_counts_stacked, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_stem)), exist_ok=True)
    np.savez(
        output_stem + '.npz',
        split=args.split,
        steps=steps,
        mse_mean=mse_mean,
        mse_std=mse_std,
        mse_per_trajectory=mse_stacked,
        n_trajectories=n_eval,
        n_at_step=n_at_step,
        sigma_percentile=args.sigma_percentile,
        radius_factor=args.radius_factor,
        mean_edges_per_step=mean_edges_per_step,
        mean_edges_by_step=mean_edges_by_step,
        edge_counts_per_trajectory=edge_counts_stacked,
    )
    summary = {
        'data_path': data_path,
        'split': args.split,
        'evaluation_provenance': provenance,
        'horizon_convention': 'forecast steps after 6-frame context',
        'mse_convention': 'mean over dynamic particles and coordinates, then trajectories',
        'model_file': model_file,
        'sigma_percentile': args.sigma_percentile,
        'radius_factor': args.radius_factor,
        'n_trajectories': n_eval,
        'max_steps': int(max_steps),
        'mse_at_final_available_step': float(mse_mean[-1]),
        'mean_edges_per_step': mean_edges_per_step,
        'mse_at_steps': {s: float(np.mean(mse_at_steps[s])) if mse_at_steps[s] else None
                         for s in EVAL_STEPS},
    }
    with open(output_stem + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-step MSE to {output_stem}.npz and {output_stem}.json")
    print("\n" + "=" * 65)
    print("Adaptive Rollout MSE")
    print(f"  sigma_percentile={args.sigma_percentile}  radius_factor={args.radius_factor}")
    print(f"  Mean edges/step: {mean_edges_per_step:.1f}")
    print("=" * 65)
    print(f"{'Step':>6}  {'MSE (mean)':>14}  {'MSE (std)':>12}")
    print("-" * 65)
    for s in EVAL_STEPS:
        vals = mse_at_steps[s]
        if vals:
            print(f"{s:>6}  {np.mean(vals):>14.6e}  {np.std(vals):>12.6e}")
        else:
            print(f"{s:>6}  (no data)")
    print("=" * 65)


if __name__ == '__main__':
    main()
