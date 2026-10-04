from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive-gns"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive-gns" / "scripts"))

from gns import learned_simulator


@pytest.fixture
def exact_radius(monkeypatch):
    """Independent tiny-graph oracle, not a substitute performance backend."""
    def radius_graph(x, r, batch, loop, max_num_neighbors):
        distances = torch.cdist(x, x)
        valid = (distances < r) & (batch[:, None] == batch[None, :])
        if not loop:
            valid.fill_diagonal_(False)
        return valid.nonzero().T.contiguous()
    monkeypatch.setattr(learned_simulator, "radius_graph", radius_graph)


@pytest.fixture
def simulator(exact_radius):
    stats = {name: {"mean": torch.zeros(2), "std": torch.ones(2)}
             for name in ("velocity", "acceleration")}
    return learned_simulator.LearnedSimulator(
        particle_dimensions=2, nnode_in=30, nedge_in=3,
        latent_dim=8, nmessage_passing_steps=1, nmlp_layers=1,
        mlp_hidden_dim=8, connectivity_radius=1.0,
        boundaries=[[0., 4.], [0., 4.]], normalization_stats=stats,
        nparticle_types=9, particle_type_embedding_size=16).eval()
