"""Simple physical allocation scores on a supplied mandatory graph.

This module does not construct a graph: caller-supplied undirected pairs retain
the graph's original numerical boundary convention. Scores use observed
displacements per stored frame and do not consume predictions or targets.
"""

import numpy as np


def physical_scores(velocities, base_pairs):
    """Return negative degree and local velocity-difference RMS scores.

    ``velocities`` must be finite real numeric values with shape ``(N, 2)``,
    ``N >= 1``. ``base_pairs`` must be an integer array with shape ``(M, 2)``
    whose unique edges obey ``0 <= i < j < N``; row order is unrestricted.

    Returned arrays have one entry per particle:
      negative_neighbor_count: float64 negative undirected degree.
      velocity_dispersion: sqrt(mean_j ||v_j - v_i||^2), float64.
      neighbor_count: integer undirected degree.
      isolated: Boolean indicator of zero degree.

    Isolated particles receive dispersion zero by definition and remain
    explicitly identifiable through ``isolated``. No clipping, graph changes,
    or fit-based fallback occurs. Nonrepresentable arithmetic raises ValueError.
    All outputs are unchanged under a global velocity offset or orthogonal
    rotation/reflection; positions are neither needed nor consulted. Inputs are
    not modified.
    """
    velocities = np.asarray(velocities)
    if velocities.dtype.kind not in "iuf":
        raise ValueError("velocities must contain real numeric values")
    if velocities.ndim != 2 or velocities.shape[1] != 2 or not len(velocities):
        raise ValueError("velocities must have nonempty shape (N, 2)")
    velocities = np.asarray(velocities, dtype=np.float64)
    if not np.isfinite(velocities).all():
        raise ValueError("velocities must contain only finite values")
    n_particles = len(velocities)

    base_pairs = np.asarray(base_pairs)
    if base_pairs.dtype.kind not in "iu":
        raise ValueError("base_pairs must have integer dtype")
    if base_pairs.ndim != 2 or base_pairs.shape[1] != 2:
        raise ValueError("base_pairs must have shape (M, 2)")
    if np.any(base_pairs < 0) or np.any(base_pairs >= n_particles):
        raise ValueError("base_pairs contain negative or out-of-bounds particle IDs")
    if np.any(base_pairs[:, 0] >= base_pairs[:, 1]):
        raise ValueError("each base pair must obey i < j; self edges are forbidden")
    if len(np.unique(base_pairs, axis=0)) != len(base_pairs):
        raise ValueError("base_pairs must not contain duplicate edges")
    base_pairs = np.asarray(base_pairs, dtype=np.int64)

    degree = np.zeros(n_particles, dtype=np.int64)
    np.add.at(degree, base_pairs[:, 0], 1)
    np.add.at(degree, base_pairs[:, 1], 1)
    isolated = degree == 0
    dispersion = np.zeros(n_particles, dtype=np.float64)
    if len(base_pairs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                differences = velocities[base_pairs[:, 1]] - velocities[base_pairs[:, 0]]
                edge_norm = np.hypot(differences[:, 0], differences[:, 1])
                scale = np.zeros(n_particles, dtype=np.float64)
                np.maximum.at(scale, base_pairs[:, 0], edge_norm)
                np.maximum.at(scale, base_pairs[:, 1], edge_norm)
                scaled_square_sum = np.zeros(n_particles, dtype=np.float64)
                for endpoint in (0, 1):
                    nodes = base_pairs[:, endpoint]
                    nonzero = scale[nodes] > 0.0
                    np.add.at(scaled_square_sum, nodes[nonzero],
                              (edge_norm[nonzero] / scale[nodes[nonzero]]) ** 2)
                connected = ~isolated
                dispersion[connected] = scale[connected] * np.sqrt(
                    scaled_square_sum[connected] / degree[connected])
        except FloatingPointError as exc:
            raise ValueError("physical scores encountered unrepresentable input arithmetic") from exc

    return {
        "negative_neighbor_count": -degree.astype(np.float64),
        "velocity_dispersion": dispersion,
        "neighbor_count": degree,
        "isolated": isolated,
    }
