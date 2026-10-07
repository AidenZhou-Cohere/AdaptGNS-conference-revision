"""Regression calibration in the same units as the acceleration training loss."""
import numpy as np


def regression_calibration(squared_error, variance, dim, n_bins=10):
    """Compare E[||error||² | v] with d v, keeping tied predictions together.

    The returned ECE has squared normalized-acceleration units. It is neither
    classification ECE nor evidence that the scalar is epistemic uncertainty.
    """
    se, var = np.asarray(squared_error), np.asarray(variance)
    if se.ndim != 1 or se.shape != var.shape or not len(se):
        raise ValueError("Expected nonempty one-dimensional matching arrays")
    if dim < 1 or n_bins < 1 or not np.isfinite(se).all() or not np.isfinite(var).all():
        raise ValueError("Calibration requires finite inputs and positive dim/bins")
    if (se < 0).any() or (var <= 0).any():
        raise ValueError("Squared errors must be nonnegative and variances positive")
    cuts = np.unique(np.quantile(var, np.linspace(0, 1, n_bins + 1)[1:-1]))
    labels = np.searchsorted(cuts, var, side="right")
    bins = []
    for label in np.unique(labels):
        keep = labels == label
        mean_se, mean_var = float(se[keep].mean()), float(var[keep].mean())
        bins.append((int(keep.sum()), mean_se, mean_var, abs(mean_se - dim * mean_var)))
    ece = sum(count * error for count, _, _, error in bins) / len(se)
    return ece, bins
