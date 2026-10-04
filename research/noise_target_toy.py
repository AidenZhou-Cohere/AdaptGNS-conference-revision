"""Verify the Gaussian noise-target derivation with synthetic CPU data only.

Run from the repository root with ``python -m research.noise_target_toy``.
This script never imports a simulator, loads benchmark data, or uses a GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def run_check():
    samples, seed = 250_000, 20261004
    tau2, s2 = 4., 1.
    delta = s2 / 5
    a = tau2 / (tau2 + delta)
    b = 1 - a
    h = tau2 * delta / (tau2 + delta)

    indices = np.arange(1, 6)
    cov_u = tau2 * np.ones((5, 5)) + delta * np.minimum.outer(indices, indices)
    cov_yu = -delta * indices
    coefficient = np.linalg.solve(cov_u, cov_yu)
    expected_coefficient = np.array([a, 0., 0., 0., -1.])
    posterior_variance = s2 - cov_yu @ coefficient
    np.testing.assert_allclose(coefficient, expected_coefficient, rtol=0., atol=1e-13)
    np.testing.assert_allclose(posterior_variance, h, rtol=0., atol=1e-13)

    rng = np.random.default_rng(seed)
    velocity = rng.normal(0., np.sqrt(tau2), samples)
    increments = rng.normal(0., np.sqrt(delta), (samples, 5))
    eta = increments.cumsum(axis=1)
    observed = velocity[:, None] + eta
    target = -eta[:, -1]
    prediction = a * observed[:, 0] - observed[:, -1]
    training_residual = target - prediction
    clean_target_residual_on_noisy_input = -prediction
    target_noise = eta[:, -1]
    clean_prediction = -b * velocity
    clean_residual = -clean_prediction

    def estimate(values, analytic):
        value = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(samples))
        result = {"analytic": float(analytic), "sample_mean": value,
                  "sample_standard_error": standard_error,
                  "difference_in_estimated_standard_errors": (value - analytic) / standard_error}
        if abs(value - analytic) > 6 * standard_error:
            raise AssertionError(result)
        return result

    checks = {
        "training_residual_second_moment": estimate(training_residual**2, h),
        "clean_residual_second_moment_at_clean_inputs": estimate(clean_residual**2, b*h),
        "clean_target_residual_second_moment_at_noisy_inputs": estimate(clean_target_residual_on_noisy_input**2, s2-h),
        "target_noise_second_moment": estimate(target_noise**2, s2),
        "residual_noise_uncentered_cross_moment": estimate(clean_target_residual_on_noisy_input*target_noise, s2-h),
    }
    expanded = (clean_target_residual_on_noisy_input**2 + target_noise**2
                - 2*clean_target_residual_on_noisy_input*target_noise)
    np.testing.assert_allclose(expanded, training_residual**2, rtol=1e-8, atol=1e-13)
    return {
        "schema": 1,
        "scope": "Synthetic Gaussian toy only; no benchmark/model data or GPU; not an estimate of a particle-experiment effect",
        "seed": seed, "samples": samples,
        "parameters": {"tau_squared": tau2, "s_squared": s2, "delta": delta},
        "analytic": {"posterior_shrinkage": a, "clean_shrinkage_bias_coefficient": -b,
                     "training_variance": h, "clean_mean_squared_error": b*h,
                     "training_variance_over_clean_mse": 1/b},
        "matrix_check": {"coefficient": coefficient.tolist(), "expected_coefficient": expected_coefficient.tolist(),
                         "posterior_variance": float(posterior_variance),
                         "maximum_coefficient_error": float(np.max(np.abs(coefficient-expected_coefficient)))},
        "monte_carlo_checks": checks,
        "decomposition_maximum_absolute_roundoff": float(np.max(np.abs(expanded-training_residual**2))),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "source": {"path": "research/noise_target_toy.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "reproduction_command": "python -m research.noise_target_toy --output research/results/noise_target_toy_check.json",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/noise_target_toy_check.json")
    args = parser.parse_args()
    result = run_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "samples": result["samples"],
                      "source_sha256": result["source"]["sha256"],
                      "matrix_maximum_coefficient_error": result["matrix_check"]["maximum_coefficient_error"],
                      "maximum_monte_carlo_standard_error_distance": max(
                          abs(value["difference_in_estimated_standard_errors"])
                          for value in result["monte_carlo_checks"].values())}, indent=2))


if __name__ == "__main__":
    main()
