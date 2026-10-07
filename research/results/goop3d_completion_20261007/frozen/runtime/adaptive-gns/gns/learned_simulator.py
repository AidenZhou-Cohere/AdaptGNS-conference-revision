import torch
import torch.nn as nn
import numpy as np
import warnings
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict
from gns.device_utils import resolve_radius_backend


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: dict,
          nparticle_types: int,
          particle_type_embedding_size: int,
          boundary_clamp_limit: float = 1.0,
          device="cpu",
          uncertainty_parameterization="variance",
          variance_floor=1e-6,
          max_num_neighbors=128,
          detach_variance_features=False,
          radius_backend="pyg",
  ):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      boundary_clamp_limit: a factor to enlarge connectivity radius used for computing
        normalized clipped distance in edge feature.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    if connectivity_radius <= 0 or variance_floor <= 0:
      raise ValueError("connectivity_radius and variance_floor must be positive")
    if uncertainty_parameterization not in {"variance", "legacy_std"}:
      raise ValueError("Unknown uncertainty parameterization")
    self._uncertainty_parameterization = uncertainty_parameterization
    self._variance_floor = variance_floor
    self._max_num_neighbors = max_num_neighbors
    radius_backend = resolve_radius_backend(radius_backend, device)
    self._radius_backend = radius_backend
    self._checkpoint_config = dict(
        particle_dimensions=particle_dimensions, nnode_in=nnode_in,
        nedge_in=nedge_in, latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps, nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim, connectivity_radius=connectivity_radius,
        boundaries=np.asarray(boundaries).tolist(),
        normalization_stats={name: {key: value.detach().cpu().clone()
                                   for key, value in stats.items()}
                             for name, stats in normalization_stats.items()},
        nparticle_types=nparticle_types,
        particle_type_embedding_size=particle_type_embedding_size,
        boundary_clamp_limit=boundary_clamp_limit,
        uncertainty_parameterization=uncertainty_parameterization,
        variance_floor=variance_floor, max_num_neighbors=max_num_neighbors,
        detach_variance_features=detach_variance_features,
        radius_backend=radius_backend)
    self._training_config = {}
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self._boundary_clamp_limit = boundary_clamp_limit

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        detach_variance_features=detach_variance_features)

    self._device = device

  def forward(self, *args, **kwargs):
    """Training entry point so DDP can install gradient synchronization hooks."""
    return self.predict_accelerations(*args, **kwargs)

  def _compute_graph_connectivity(
          self,
          node_features: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius

    Args:
      node_features: Node features with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Specify examples id for particles
    counts = torch.as_tensor(nparticles_per_example, device=node_features.device,
                             dtype=torch.long).flatten()
    if bool((counts <= 0).any()) or int(counts.sum()) != len(node_features):
      raise ValueError("Particle counts must be positive and sum to node count")

    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    if self._radius_backend in {"scipy", "scipy_host"}:
      if node_features.device.type != "cpu":
        if self._radius_backend != "scipy_host":
          raise ValueError("scipy is CPU-only; use scipy_host for explicit host graph transfer")
      from scipy.spatial import cKDTree
      points = node_features.detach().cpu().numpy()
      sources, targets, offset = [], [], 0
      for count in counts.cpu().tolist():
        local = points[offset:offset + count]
        tree = cKDTree(local)
        for target, neighbors in enumerate(tree.query_ball_point(local, radius)):
          # Match PyG's strict-radius rule. When capped, choose nearest neighbors
          # deterministically; PyG's cap can choose a different subset.
          neighbors = np.asarray(neighbors, dtype=np.int64)
          distances = np.linalg.norm(local[neighbors] - local[target], axis=1)
          keep = distances < radius
          if not add_self_edges:
            keep &= neighbors != target
          neighbors, distances = neighbors[keep], distances[keep]
          order = np.lexsort((neighbors, distances))[:self._max_num_neighbors]
          sources.extend((neighbors[order] + offset).tolist())
          targets.extend([target + offset] * len(order))
        offset += count
      edge_index = torch.tensor([sources, targets], dtype=torch.long,
                                device=node_features.device)
    else:
      batch_ids = torch.repeat_interleave(
          torch.arange(len(counts), device=node_features.device), counts)
      edge_index = radius_graph(
          node_features, r=radius, batch=batch_ids, loop=add_self_edges,
          max_num_neighbors=self._max_num_neighbors)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    senders = edge_index[0, :]
    receivers = edge_index[1, :]
    return senders, receivers

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None,
          augment_radius_prob: float = 0.0,
          augment_radius_factor: float = 1.267):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles)
      augment_radius_prob: If > 0 and self.training is True, randomly selects this
        fraction of particles and gives them edges out to augment_radius_factor*r.
        Mirrors the inference-time adaptive policy with random selection so the
        model learns to handle mixed-radius graphs.
      augment_radius_factor: Multiplier on connectivity_radius for selected particles.
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence)

    # Get connectivity of the graph with shape of (nparticles, 2)
    if self.training and augment_radius_prob > 0.0:
      high_radius_mask = (
          torch.rand(nparticles, device=most_recent_position.device)
          < augment_radius_prob)
      r_large = self._connectivity_radius * augment_radius_factor
      senders, receivers = self._build_adaptive_edge_index(
          most_recent_position, nparticles_per_example,
          high_radius_mask, r_large)
    else:
      senders, receivers = self._compute_graph_connectivity(
          most_recent_position, nparticles_per_example, self._connectivity_radius)
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(
        nparticles, -1)
    # There are 5 previous steps, with dim 2
    # node_features shape (nparticles, 5 * 2 = 10)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to lower and upper boundaries.
    # boundaries are an array of shape [num_dimensions, 2], where the second
    # axis, provides the lower/upper boundaries.
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    # The distance to 4 boundaries (top/bottom/left/right)
    # node_features shape (nparticles, 10+4)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 30) for 2D (if material_property is not valid in training example)
    # 30 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding

    # Material property
    if material_property is not None:
        material_property = material_property.view(nparticles, 1)
        node_features.append(material_property)
    # Final node_features shape (nparticles, 31) for 2D
    # 31 = 10 (5 velocity sequences*dim) + 4 boundaries + 16 particle embedding + 1 material property

    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    # normalized_relative_displacements = (
    #     torch.gather(most_recent_position, 0, senders) -
    #     torch.gather(most_recent_position, 0, receivers)
    # ) / self._connectivity_radius
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius

    # Add relative displacement between two particles as an edge feature
    # with shape (nparticles, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nparticles, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles)

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
    predicted_normalized_acceleration, _ = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions

  def predict_positions_with_variance(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None):
    """Predict position and positive uncertainty-head output.

    Returns:
      next_positions (torch.tensor): Next position of particles.
      variance (torch.tensor): Head output, shape (nparticles,). It is variance
        for corrected objectives and std for legacy checkpoints. Convert with
        head_to_variance before calibration in normalized acceleration units.
    """
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types, material_property)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
    predicted_normalized_acceleration, variance = self._encode_process_decode(
        node_features, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions, variance

  def _build_adaptive_edge_index(
          self,
          most_recent_position: torch.tensor,
          nparticles_per_example: torch.tensor,
          high_sigma_mask: torch.tensor,
          r_large: float):
    """Build a combined edge index: base edges (radius r) plus extended edges
    (radius r_large) filtered to those touching at least one high-sigma particle.

    Returns senders, receivers in the same (edge_index[0], edge_index[1]) convention
    used throughout _encoder_preprocessor.
    """
    if r_large < self._connectivity_radius:
      raise ValueError("Expanded radius cannot be smaller than base radius")
    base_s, base_r = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    if r_large == self._connectivity_radius or not bool(high_sigma_mask.any()):
      return base_s, base_r
    ext_s, ext_r = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, r_large)

    # Keep extended edges only where at least one endpoint is high-sigma
    keep = high_sigma_mask[ext_s] | high_sigma_mask[ext_r]
    ext_s = ext_s[keep]
    ext_r = ext_r[keep]

    all_s = torch.cat([base_s, ext_s])
    all_r = torch.cat([base_r, ext_r])

    combined = torch.stack([all_s, all_r], dim=0)  # (2, n_edges)
    # Metal does not provide every unique(dim=...) variant. Graph set operations
    # remain explicitly on the host for the hybrid backend, not a hidden kernel fallback.
    if combined.device.type == "mps":
      combined = torch.unique(combined.cpu(), dim=1).to(most_recent_position.device)
    else:
      combined = torch.unique(combined, dim=1)
    return combined[0], combined[1]

  def _forward_with_edge_index(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          senders: torch.tensor,
          receivers: torch.tensor,
          material_property: torch.tensor = None):
    """Run the GNN forward pass with a pre-built edge index.

    senders / receivers follow PyG's source_to_target convention:
    edge_index[0] is source, edge_index[1] is target.
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]
    velocity_sequence = time_diff(position_sequence)

    node_features = []

    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
    node_features.append(flat_velocity_sequence)

    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
    distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    node_features.append(normalized_clipped_distance_to_boundaries)

    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(particle_types)
      node_features.append(particle_type_embeddings)

    if material_property is not None:
      node_features.append(material_property.view(nparticles, 1))

    node_features_cat = torch.cat(node_features, dim=-1)

    # Edge features (same normalization as _encoder_preprocessor)
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features = torch.cat(
        [normalized_relative_displacements, normalized_relative_distances], dim=-1)

    edge_index = torch.stack([senders, receivers])

    predicted_normalized_acceleration, variance = self._encode_process_decode(
        node_features_cat, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, position_sequence)
    return next_positions, variance

  def predict_positions_adaptive(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          sigma_prev: torch.tensor,
          sigma_percentile: float = 80.0,
          radius_factor: float = 1.267,
          material_property: torch.tensor = None):
    """Single-pass adaptive rollout step using sigma from the *previous* step.

    The graph topology is decided using sigma_prev (already known), so only ONE
    GNN forward pass is needed. The two graph searches, union and larger graph
    add overhead that must be timed separately from the neural forward pass.

    High-uncertainty particles (sigma_prev > sigma_percentile-th percentile) receive
    extra edges out to radius_factor * base_radius. Either selected endpoint
    admits the edge, so unselected particles can also gain neighbors. At a
    common state, base edges are preserved; counts across different rollouts
    are additionally affected by the predicted geometry.

    Args:
      current_positions: Position history (nparticles, history_len, dim).
      nparticles_per_example: Particles per batch element.
      particle_types: Particle types (nparticles,).
      sigma_prev: Per-particle sigma from the previous rollout step, shape (nparticles,).
      sigma_percentile: Percentile threshold; particles above this get expanded edges.
      radius_factor: Multiplier for the expanded connectivity radius.
      material_property: Optional per-particle material property.

    Returns:
      next_positions: Predicted positions, shape (nparticles, dim).
      sigma_curr: Per-particle sigma from this step, shape (nparticles,).
    """
    high_sigma_mask = self._select_high_uncertainty(
        sigma_prev, nparticles_per_example, sigma_percentile)

    most_recent_position = current_positions[:, -1]
    r_large = self._connectivity_radius * radius_factor

    senders, receivers = self._build_adaptive_edge_index(
        most_recent_position, nparticles_per_example, high_sigma_mask, r_large)

    next_positions, sigma_curr = self._forward_with_edge_index(
        current_positions, nparticles_per_example, particle_types,
        senders, receivers, material_property)

    return next_positions, sigma_curr

  @staticmethod
  def _select_high_uncertainty(scores, nparticles_per_example, percentile):
    """Threshold independently in each trajectory; ties are not expanded.

    This preserves the released strict-quantile policy for a single trajectory.
    It is not an exact edge or particle budget, particularly when scores tie.
    """
    if not 0 <= percentile <= 100:
      raise ValueError("percentile must lie in [0, 100]")
    counts = [int(n) for n in nparticles_per_example]
    if any(n <= 0 for n in counts) or sum(counts) != scores.numel():
      raise ValueError("Particle counts must be positive and match scores")
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
      raise ValueError("Uncertainty scores must be a finite one-dimensional tensor")
    output_device = scores.device
    if scores.device.type == "mps":
      # The controller is discrete; host quantiles preserve the CPU policy and
      # avoid relying on unsupported Metal sort/quantile kernels.
      scores = scores.detach().cpu()
    mask = torch.zeros_like(scores, dtype=torch.bool)
    start = 0
    for count in counts:
      local = scores[start:start + count]
      mask[start:start + count] = local > torch.quantile(local, percentile / 100)
      start += count
    return mask.to(output_device)

  def head_to_variance(self, head):
    """Convert head output to per-coordinate normalized acceleration variance."""
    if self._uncertainty_parameterization == "legacy_std":
      return head.square() + self._variance_floor
    return head.clamp_min(self._variance_floor)

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor,
          material_property: torch.tensor = None,
          augment_radius_prob: float = 0.0,
          augment_radius_factor: float = 1.267):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles).
      augment_radius_prob: Per-particle probability of receiving expanded radius
        during training (graph augmentation). Pass-through to _encoder_preprocessor.
      augment_radius_factor: Multiplier on connectivity_radius for augmented particles.

    Returns:
      Predicted normalized acceleration (N,d), positive raw head output (N,),
        and target normalized acceleration (N,d), in that order.

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    if material_property is not None:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types,
            material_property,
            augment_radius_prob=augment_radius_prob,
            augment_radius_factor=augment_radius_factor)
    else:
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types,
            augment_radius_prob=augment_radius_prob,
            augment_radius_factor=augment_radius_factor)
    predicted_normalized_acceleration, predicted_variance = self._encode_process_decode(
        node_features, edge_index, edge_features)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, predicted_variance, target_normalized_acceleration

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save({"format_version": 2, "state_dict": self.state_dict(),
                "simulator_config": self._checkpoint_config,
                "training_config": self._training_config}, path)

  def load(
          self,
          path: str,
          allow_missing_variance_head: bool = False):
    """Load model state from file

    Args:
      path: Model path
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
      config = payload["simulator_config"]
      for key in ("connectivity_radius", "uncertainty_parameterization",
                  "variance_floor", "max_num_neighbors"):
        if self._checkpoint_config[key] != config[key]:
          raise ValueError(f"Checkpoint {key} differs from simulator configuration; "
                           "load with gns.model_io.load_for_evaluation")
      for name, stats in config["normalization_stats"].items():
        self._normalization_stats[name] = {
            key: value.to(self._device) for key, value in stats.items()}
      self._checkpoint_config = config
      self._training_config = payload.get("training_config", {})
      state = payload["state_dict"]
    else:
      warnings.warn("Legacy checkpoint omits normalization, architecture and loss "
                    "metadata; verify these settings against the training recipe.",
                    UserWarning)
      state = payload
    if allow_missing_variance_head:
      incompatible = self.load_state_dict(state, strict=False)
      allowed = "_encode_process_decode._variance_head."
      if incompatible.unexpected_keys or any(
          not key.startswith(allowed) for key in incompatible.missing_keys):
        raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible}")
      if incompatible.missing_keys:
        warnings.warn("Upstream mean-only checkpoint: variance head is untrained; "
                      "use only fixed-graph mean evaluation.", UserWarning)
    else:
      self.load_state_dict(state)


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
