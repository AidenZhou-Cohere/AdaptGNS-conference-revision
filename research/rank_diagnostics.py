"""Within-frame rank diagnostics with explicit undefined outcomes.

These CPU-only helpers neither pool particles across frames nor compute
p-values. Partial rank correlation is a descriptive linear adjustment of
average ranks; it is not a causal adjustment or a conditional-independence test.
"""

import numpy as np
from scipy.stats import rankdata


def _finite_real_array(value, name, ndim):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    # Preserve integer order before ranking, including integers above 2**53.
    return array


def spearman_record(x, y, controls=None):
    """Return ``value``, ``reason`` and ``n`` for matching finite real vectors.

    With no controls, the value is Pearson correlation of centered average
    ranks (ordinary tie-aware Spearman correlation). Malformed/nonfinite inputs
    raise ``ValueError``; fewer than three observations and constant vectors
    return a null value with an explicit reason. No observations are dropped.

    ``controls`` must have shape ``(n, k)``; ``k=0`` is permitted. Both rank
    vectors are residualized against an intercept and the average ranks of
    each control using ``np.linalg.lstsq(..., rcond=1e-12)``. Constant/collinear
    controls are permitted. The result records ``control_design_rank`` and
    ``residual_degrees_of_freedom = n - control_design_rank``, even when its
    coefficient is undefined. At least three residual degrees of freedom are
    required. A residual norm no greater than 1e-10 times that vector's
    original centered-rank norm is undefined, preventing roundoff from being
    reported as a residual relationship. An explicit empty control matrix
    still fits an intercept and therefore uses one residual degree of freedom.

    Inputs are not modified. Finite coefficients are clipped to [-1, 1] only
    to absorb numerical error in the normalized dot product.
    """
    x = _finite_real_array(x, "x", 1)
    y = _finite_real_array(y, "y", 1)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    n = int(x.size)
    if controls is not None:
        controls = _finite_real_array(controls, "controls", 2)
        if controls.shape[0] != n:
            raise ValueError("controls must have shape (n, k) matching x and y")

    ranks = np.column_stack((rankdata(x, method="average"),
                             rankdata(y, method="average")))
    result = {"value": None, "reason": None, "n": n}
    residuals = None
    if controls is not None:
        control_ranks = rankdata(controls, method="average", axis=0)
        design = np.column_stack((np.ones(n), control_ranks))
        coefficients, _, rank, _ = np.linalg.lstsq(design, ranks, rcond=1e-12)
        residuals = ranks - design @ coefficients
        result["control_design_rank"] = int(rank)
        result["residual_degrees_of_freedom"] = n - int(rank)

    if n < 3:
        result["reason"] = "fewer_than_three_particles"
        return result

    centered = ranks - ranks.mean(axis=0)
    original_norms = np.linalg.norm(centered, axis=0)
    for index, name in enumerate(("x", "y")):
        if original_norms[index] == 0.0:
            result["reason"] = f"constant_{name}"
            return result

    vectors = centered
    norms = original_norms
    if controls is not None:
        if result["residual_degrees_of_freedom"] < 3:
            result["reason"] = "insufficient_residual_degrees_of_freedom"
            return result
        # The fitted intercept makes this zero in exact arithmetic. Recenter
        # before Pearson correlation to remove its floating-point remainder.
        vectors = residuals - residuals.mean(axis=0)
        norms = np.linalg.norm(vectors, axis=0)
        for index, name in enumerate(("x", "y")):
            if norms[index] <= 1e-10 * original_norms[index]:
                result["reason"] = f"negligible_{name}_residual"
                return result

    normalized = vectors / norms
    result["value"] = float(np.clip(np.dot(normalized[:, 0], normalized[:, 1]),
                                    -1.0, 1.0))
    return result
