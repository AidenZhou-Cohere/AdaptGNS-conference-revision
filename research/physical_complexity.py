"""Observed-state, two-dimensional local flow diagnostics.

These are physical proxies, not a definition or validation of model uncertainty.
Velocities are the last observed displacement per stored frame; derivatives
therefore have inverse-stored-frame units. No target or forecast is required.
The implementation is CPU-only and does not read models, files, or datasets.
"""

from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


def _state_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if array.ndim != 2 or array.shape[1] != 2 or not len(array):
        raise ValueError(f"{name} must have nonempty shape (N, 2)")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_scalar(value, name, minimum=0.0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0 or value < minimum:
        raise ValueError(f"{name} must be positive and at least {minimum}")
    return value


def local_flow_diagnostics(positions, velocities, radius=0.015,
                         min_neighbors=3, max_condition=1e6):
    """Return local kinematic proxies from matching finite ``(N, 2)`` arrays.

    Neighbors have strict float64 squared Euclidean distance less than
    ``radius**2``. Self IDs are excluded; distinct particles at identical
    coordinates remain neighbors. Sorted IDs give deterministic summation.

    For particle i, fit ``DV = DX @ G.T`` to unweighted neighbor differences,
    scaling DX by radius before conditioning and fitting. ``fit_condition``
    is the eigenvalue ratio of the scaled Gram matrix (the square of the
    design-matrix condition). A fit requires at least ``min_neighbors``,
    numerical rank two, and condition at most ``max_condition``. The count
    threshold may be any positive integer; rank two is always required.

    Returns a dict of per-particle arrays:
      neighbor_count: number of strict-radius neighbors, integer.
      speed: Euclidean norm of the supplied velocity.
      velocity_dispersion: sqrt(mean_j ||v_j - v_i||^2).
      gradient: G, shape (N, 2, 2), with G[a,b] = d v_a / d x_b.
      strain_frobenius: ||(G + G.T)/2||_F, including isotropic expansion.
      abs_vorticity: |G[1,0] - G[0,1]|.
      abs_divergence: |trace(G)|.
      fit_valid: Boolean fit eligibility.
      fit_condition: Gram condition, infinity if rank deficient and NaN
        when there are no neighbors. It may be finite for a neighborhood
        rejected by the separate neighbor-count threshold.

    Undefined dispersion (no neighbors) and gradient-derived proxies (invalid
    fit) are NaN. They are never clipped or imputed. Unrepresentable numeric
    operations raise ValueError rather than silently returning corrupt values.
    Inputs are not modified.
    """
    positions = _state_array(positions, "positions")
    velocities = _state_array(velocities, "velocities")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    radius = _positive_scalar(radius, "radius")
    radius_squared = radius * radius
    if not np.isfinite(radius_squared) or radius_squared == 0.0:
        raise ValueError("radius squared must be finite and nonzero in float64")
    max_condition = _positive_scalar(max_condition, "max_condition", minimum=1.0)
    if (isinstance(min_neighbors, (bool, np.bool_))
            or not isinstance(min_neighbors, Integral) or min_neighbors < 1):
        raise ValueError("min_neighbors must be a positive integer")

    n_particles = len(positions)
    result = {
        "neighbor_count": np.zeros(n_particles, dtype=np.int64),
        "speed": np.empty(n_particles, dtype=np.float64),
        "velocity_dispersion": np.full(n_particles, np.nan),
        "gradient": np.full((n_particles, 2, 2), np.nan),
        "strain_frobenius": np.full(n_particles, np.nan),
        "abs_vorticity": np.full(n_particles, np.nan),
        "abs_divergence": np.full(n_particles, np.nan),
        "fit_valid": np.zeros(n_particles, dtype=bool),
        "fit_condition": np.full(n_particles, np.nan),
    }
    tree = cKDTree(positions)
    # A Chebyshev query supplies a conservative candidate box. The float64
    # squared-Euclidean recheck below alone defines actual membership.
    candidate_radius = np.nextafter(radius, np.inf)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result["speed"][:] = np.hypot(velocities[:, 0], velocities[:, 1])
            for i in range(n_particles):
                candidates = np.asarray(tree.query_ball_point(
                    positions[i], candidate_radius, p=np.inf,
                    return_sorted=True), dtype=np.int64)
                candidates = candidates[candidates != i]
                dx = positions[candidates] - positions[i]
                keep = np.sum(dx * dx, axis=1) < radius_squared
                neighbors, dx = candidates[keep], dx[keep]
                count = len(neighbors)
                result["neighbor_count"][i] = count
                if count == 0:
                    continue
                dv = velocities[neighbors] - velocities[i]
                # Scaling avoids overflow when squaring otherwise representable
                # differences; it does not change the mathematical RMS.
                scale = np.max(np.abs(dv))
                result["velocity_dispersion"][i] = (
                    0.0 if scale == 0 else scale * np.sqrt(
                        np.mean(np.sum((dv / scale) ** 2, axis=1))))
                design = dx / radius
                eigenvalues = np.linalg.eigvalsh(design.T @ design)
                rank = np.linalg.matrix_rank(design)
                if rank < 2 or eigenvalues[0] <= 0.0:
                    result["fit_condition"][i] = np.inf
                    continue
                condition = eigenvalues[-1] / eigenvalues[0]
                result["fit_condition"][i] = condition
                if count < min_neighbors or condition > max_condition:
                    continue
                solution, _, rank, _ = np.linalg.lstsq(design, dv, rcond=None)
                if rank != 2:
                    raise ValueError("inconsistent numerical rank in affine fit")
                gradient = solution.T / radius
                strain = 0.5 * gradient + 0.5 * gradient.T
                strain_norm = np.hypot.reduce(strain.ravel())
                vorticity = abs(gradient[1, 0] - gradient[0, 1])
                divergence = abs(gradient[0, 0] + gradient[1, 1])
                if not np.isfinite(gradient).all():
                    raise ValueError("unrepresentable affine gradient")
                result["gradient"][i] = gradient
                result["strain_frobenius"][i] = strain_norm
                result["abs_vorticity"][i] = vorticity
                result["abs_divergence"][i] = divergence
                result["fit_valid"][i] = True
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("local flow diagnostic encountered unrepresentable input arithmetic") from exc
    return result
