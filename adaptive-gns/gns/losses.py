"""Acceleration objectives; uncertainty lives in normalized acceleration space."""
import torch


def acceleration_loss(pred_acc, target_acc, non_kinematic_mask,
                      pred_variance=None, loss_type="nll", variance_floor=1e-6):
    """Average over dynamic particles, summing acceleration dimensions.

    In ``nll`` mode the Softplus head predicts isotropic variance v, and the
    constant-free Gaussian NLL is ||e||²/(2v) + d log(v)/2. ``legacy_nll``
    exactly retains the released implementation: v = head² + floor, and
    log(v)/2 (missing d). It is for provenance checks, not a Gaussian NLL
    in d > 1. MSE ignores the uncertainty head. ``faithful`` combines the MSE
    mean objective with detached-residual NLL and requires detached head inputs.
    """
    if loss_type not in {"nll", "legacy_nll", "mse", "faithful"}:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    mask = non_kinematic_mask.bool()
    if not bool(mask.any()):
        raise ValueError("Acceleration loss requires a dynamic particle")
    # Index before arithmetic so NaNs on prescribed kinematic states cannot
    # contaminate this loss or its gradients.
    residual = pred_acc[mask] - target_acc[mask]
    squared_error = residual.square().sum(dim=-1)
    if pred_variance is None or loss_type == "mse":
        return squared_error.mean()
    head = pred_variance[mask]
    if loss_type == "legacy_nll":
        variance = head.square() + variance_floor
        log_coefficient = 0.5
    else:
        variance = head.clamp_min(variance_floor)
        log_coefficient = 0.5 * pred_acc.shape[-1]
    if loss_type == "faithful":
        # Requires a detached input to the variance head as well. The mean term
        # deliberately matches this module's MSE convention (sum over d).
        return (squared_error + squared_error.detach() / (2 * variance)
                + log_coefficient * variance.log()).mean()
    return (squared_error / (2 * variance)
            + log_coefficient * variance.log()).mean()
