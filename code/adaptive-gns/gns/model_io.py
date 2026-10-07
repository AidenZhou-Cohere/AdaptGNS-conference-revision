"""Simulator construction and explicit reconstruction of legacy checkpoints."""
import copy
import warnings

import torch

from gns.learned_simulator import LearnedSimulator
from gns.device_utils import resolve_radius_backend, runtime_provenance


def build_simulator(metadata, acc_noise_std, vel_noise_std, device,
                    connectivity_radius=None, nmessage_passing_steps=10,
                    uncertainty_parameterization="variance", variance_floor=1e-6,
                    detach_variance_features=False, radius_backend="pyg"):
    normalization = {}
    for name, prefix, noise in (("acceleration", "acc", acc_noise_std),
                                ("velocity", "vel", vel_noise_std)):
        normalization[name] = {
            "mean": torch.tensor(metadata[prefix + "_mean"], dtype=torch.float32,
                                 device=device),
            "std": torch.sqrt(torch.tensor(metadata[prefix + "_std"],
                                           dtype=torch.float32, device=device).square()
                              + noise ** 2),
        }
    return LearnedSimulator(
        particle_dimensions=metadata["dim"],
        nnode_in=metadata.get("nnode_in", 7 * metadata["dim"] + 16),
        nedge_in=metadata.get("nedge_in", metadata["dim"] + 1),
        latent_dim=128, nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=2, mlp_hidden_dim=128,
        connectivity_radius=(metadata["default_connectivity_radius"]
                             if connectivity_radius is None else connectivity_radius),
        boundaries=metadata["bounds"], normalization_stats=normalization,
        nparticle_types=9, particle_type_embedding_size=16,
        boundary_clamp_limit=metadata.get("boundary_augment", 1.0),
        device=device, uncertainty_parameterization=uncertainty_parameterization,
        variance_floor=variance_floor, detach_variance_features=detach_variance_features,
        radius_backend=radius_backend)


def load_for_evaluation(path, metadata, device, normalization_noise_std=None,
                        connectivity_radius=None, nmessage_passing_steps=None,
                        allow_missing_variance_head=False, radius_backend=None):
    """Load saved architecture and training normalization; do not inject noise.

    A legacy state_dict needs its original noise-normalization setting. If
    omitted, use 6.7e-4, the released training default, with an explicit warning.
    Passing zero is available only to reproduce the old evaluator mismatch.
    Radius overrides alter graph, boundary and edge-feature normalization,
    exactly as a simulator trained at that radius would; report this intervention.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "simulator_config" in payload:
        if (normalization_noise_std is not None or nmessage_passing_steps is not None):
            raise ValueError("Configured checkpoints already store normalization and "
                             "architecture; legacy reconstruction overrides are invalid")
        config = copy.deepcopy(payload["simulator_config"])
        checkpoint_radius_backend = config.get("radius_backend", "pyg")
        # Graph backend is an explicit runtime override, unlike architecture or
        # normalization. Record both values because neighbor caps may differ.
        config["radius_backend"] = resolve_radius_backend(
            checkpoint_radius_backend if radius_backend is None else radius_backend, device)
        config["normalization_stats"] = {
            name: {key: value.to(device) for key, value in stats.items()}
            for name, stats in config["normalization_stats"].items()}
        simulator = LearnedSimulator(**config, device=device)
        simulator.load(path, allow_missing_variance_head=allow_missing_variance_head)
        simulator._checkpoint_config = copy.deepcopy(simulator._checkpoint_config)
        simulator._checkpoint_config["radius_backend"] = simulator._radius_backend
        if (payload.get("training_config", {}).get("loss") == "mse"
                and not allow_missing_variance_head):
            raise ValueError("MSE checkpoint has an untrained variance head; use "
                             "fixed-graph mean evaluation, not risk allocation/calibration")
        provenance = {"checkpoint_format": 2,
                      "normalization_source": "checkpoint",
                      "checkpoint_radius_backend": checkpoint_radius_backend,
                      "training_config": payload.get("training_config", {})}
    else:
        noise = 6.7e-4 if normalization_noise_std is None else normalization_noise_std
        if normalization_noise_std is None:
            warnings.warn("Assuming released training normalization noise_std=6.7e-4 "
                          "for a legacy checkpoint; override if its recipe differs.")
        simulator = build_simulator(
            metadata, noise, noise, device,
            connectivity_radius=connectivity_radius,
            nmessage_passing_steps=(10 if nmessage_passing_steps is None
                                    else nmessage_passing_steps),
            uncertainty_parameterization="legacy_std",
            radius_backend=resolve_radius_backend(radius_backend, device))
        simulator.load(path, allow_missing_variance_head=allow_missing_variance_head)
        provenance = {"checkpoint_format": "legacy_state_dict",
                      "normalization_source": "explicit_or_assumed_training_recipe",
                      "normalization_noise_std": noise}
    if connectivity_radius is not None:
        simulator._connectivity_radius = connectivity_radius
    simulator.to(device).eval()
    provenance.update({
        "connectivity_radius": simulator._connectivity_radius,
        "nmessage_passing_steps": len(simulator._encode_process_decode._processor.gnn_stacks),
        "uncertainty_parameterization": simulator._uncertainty_parameterization,
        "variance_floor": simulator._variance_floor,
        "max_num_neighbors": simulator._max_num_neighbors,
        "radius_backend": simulator._radius_backend,
        "edge_convention": "directed_with_self_loops",
        "runtime": runtime_provenance(device, simulator._radius_backend),
    })
    return simulator, provenance
