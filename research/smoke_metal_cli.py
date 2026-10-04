#!/usr/bin/env python3
"""Bounded learning/integration check through the original GNS command-line tools.

Uses two official pilot training trajectories, a short validation segment, the
full128x10 architecture, and matched CPU/MPS seeds. It is not a benchmark result
for the paper and never reads the test split.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "adaptive-gns"))
from gns.device_utils import resolve_device, synchronize
from gns.losses import acceleration_loss
from gns.model_io import build_simulator, load_for_evaluation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(source_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for split, count in (("train", 2), ("valid", 1)):
        source = source_dir / f"{split}-pilot.npz"
        with np.load(source, allow_pickle=False) as archive:
            entries = np.empty(count, dtype=object)
            for index in range(count):
                positions = archive[f"position_{index}"]
                if split == "valid":
                    positions = positions[450:466]
                kinds = archive[f"type_{index}"]
                entries[index] = (positions.copy(), kinds.copy())
        # The legacy original CLI format is generated locally from trusted
        # numeric arrays solely for this bounded compatibility check.
        np.savez(destination / f"{split}.npz", gns_data=entries)
        provenance[split] = {"source_sha256": digest(source), "examples": count,
                             "recorded_frame_slice": [450, 466] if split == "valid" else "all",
                             "fixture_sha256": digest(destination / f"{split}.npz")}
    metadata = json.loads((source_dir / "metadata.json").read_text())
    (destination / "metadata.json").write_text(json.dumps(metadata))
    return metadata, provenance


def run_logged(command, directory, log):
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0", OMP_NUM_THREADS="1",
               PYTHONWARNINGS="ignore::SyntaxWarning")
    start = time.perf_counter()
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


@torch.no_grad()
def validation_metrics(model, positions, kinds, device):
    mse, nll = [], []
    for frame in range(6, len(positions)):
        history = torch.as_tensor(np.ascontiguousarray(positions[frame-6:frame].transpose(1, 0, 2)), device=device)
        labels = torch.as_tensor(positions[frame], device=device)
        types = torch.as_tensor(kinds, dtype=torch.long, device=device)
        pred, head, target = model(next_positions=labels,
            position_sequence_noise=torch.zeros_like(history), position_sequence=history,
            nparticles_per_example=[len(kinds)], particle_types=types)
        mask = types != 3
        mse.append(float((pred[mask]-target[mask]).square().sum(-1).mean().cpu()))
        nll.append(float(acceleration_loss(pred, target, mask, head, loss_type="nll").cpu()))
    synchronize(device)
    return {"normalized_acceleration_vector_mse": float(np.mean(mse)),
            "corrected_nll": float(np.mean(nll)), "validation_states": len(mse)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/metal_cli_smoke.json")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    if not 1 <= args.updates <= 1000:
        parser.error("This integration smoke is bounded to1..1000updates")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    resolve_device("mps")
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata, provenance = make_fixture(args.source_data.resolve(), root / "data")
    with np.load(args.source_data / "valid-pilot.npz", allow_pickle=False) as z:
        validation_position, validation_kinds = z["position_0"][450:466], z["type_0"]
    result = {"kind": "bounded_original_cli_cpu_mps_learning_sanity", "updates_per_device": args.updates,
              "seed": args.seed, "loss": "faithful", "architecture": {"latent_dim": 128, "message_passing_steps": 10, "mlp_hidden_layers": 2},
              "test_split_used": False, "mps_cpu_fallback": "0", "fixture_provenance": provenance,
              "cautions": ["100 updates are an integration check, not evidence of convergence or conference-level accuracy.",
                           "CPU/Metal stochastic optimization paths need not be identical; original strict gradient-parity caveats remain.",
                           "Validation uses one short contiguous segment, selected for a bounded check, not a full representative validation set."],
              "devices": {}}
    for device in ("cpu", "mps"):
        directory = root / device
        model_dir = directory / "models"
        if model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite an existing smoke checkpoint: {model_dir}")
        model_dir.mkdir(parents=True)
        command = [sys.executable, "-m", "gns.train", "--device="+device,
                   "--radius_backend=scipy_host", "--cpu_threads=1", "--loss=faithful",
                   "--data_path="+str(root / "data")+"/", "--model_path="+str(model_dir)+"/",
                   "--batch_size=2", "--nmessage_passing_steps=10", "--ntraining_steps="+str(args.updates),
                   "--nsave_steps="+str(args.updates), "--validation_interval=25", "--seed="+str(args.seed)]
        print(f"Starting bounded original CLI: {device}, {args.updates}updates", flush=True)
        wall = run_logged(command, REPO / "adaptive-gns", directory / "train.log")
        text = (directory / "train.log").read_text()
        losses = [float(value) for value in re.findall(r"step = \d+/\d+, loss = ([^\s]+)", text)]
        if len(losses) != args.updates or not np.isfinite(losses).all():
            raise RuntimeError(f"Expected{args.updates}finite training losses, got{len(losses)}on{device}")
        checkpoint = model_dir / f"model-{args.updates}.pt"
        state_path = model_dir / f"train_state-{args.updates}.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["global_train_state"]["step"] != args.updates:
            raise RuntimeError("Checkpoint update counter mismatch")
        torch.manual_seed(args.seed)
        initial = build_simulator(metadata, 6.7e-4, 6.7e-4, device,
                                  detach_variance_features=True,
                                  radius_backend="scipy_host").to(device).eval()
        before = validation_metrics(initial, validation_position, validation_kinds, device)
        del initial
        trained, load_provenance = load_for_evaluation(checkpoint, metadata, device)
        after = validation_metrics(trained, validation_position, validation_kinds, device)
        del trained
        result["devices"][device] = {"training_command": command, "wall_seconds_including_startup_validation_and_save": wall,
            "training_losses": losses, "first_10_mean_loss": float(np.mean(losses[:10])), "last_10_mean_loss": float(np.mean(losses[-10:])),
            "validation_before": before, "validation_after": after, "checkpoint_sha256": digest(checkpoint),
            "train_state_sha256": digest(state_path), "checkpoint_path": str(checkpoint), "load_provenance": load_provenance}
        print(f"{device}: wall={wall:.2f}s; validation vectorMSE {before['normalized_acceleration_vector_mse']:.6g} -> {after['normalized_acceleration_vector_mse']:.6g}", flush=True)
        if device == "mps":
            for script in ("evaluate_rollout_mse.py", "evaluate_adaptive_rollout.py"):
                output_stem = directory / script.removesuffix(".py")
                evaluation = [sys.executable, str(REPO / "adaptive-gns/scripts" / script),
                              "--device=mps", "--radius_backend=scipy_host", "--split=valid",
                              "--data_path="+str(root / "data"), "--model_path="+str(model_dir),
                              "--model_file="+checkpoint.name, "--max_trajectories=1", "--output="+str(output_stem)]
                duration = run_logged(evaluation, REPO, output_stem.with_suffix(".log"))
                with np.load(output_stem.with_suffix(".npz"), allow_pickle=False) as z:
                    if z["mse_per_trajectory"].shape != (1, 10) or not np.isfinite(z["mse_per_trajectory"]).all():
                        raise RuntimeError(f"Invalid bounded rollout from{script}")
                result["devices"][device].setdefault("evaluation_cli_checks", []).append({"script": script, "predicted_steps": 10, "finite": True, "wall_seconds": duration})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    cpu = np.asarray(result["devices"]["cpu"]["training_losses"])
    mps = np.asarray(result["devices"]["mps"]["training_losses"])
    result["paired_loss_curve_comparison"] = {"maximum_absolute_difference": float(np.max(np.abs(cpu-mps))),
        "relative_l2_difference": float(np.linalg.norm(cpu-mps)/np.linalg.norm(cpu)),
        "description": "Matched seeds and shuffled examples; not a claim of bitwise optimization equivalence"}
    result["source_code_sha256"] = {name: digest(REPO / "adaptive-gns/gns" / name)
        for name in ("device_utils.py", "learned_simulator.py", "train.py", "model_io.py")}
    result["smoke_script_sha256"] = digest(__file__)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
