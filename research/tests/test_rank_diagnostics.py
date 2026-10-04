import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from research.rank_diagnostics import spearman_record


@pytest.mark.parametrize("x,y", [
    ([1, 1, 3, 2, 5, 5], [4, 2, 2, 1, 3, 6]),
    ([4, 3, 2, 1], [1, 2, 3, 4]),
    ([1, 2, 3], [1, 2, 3]),
    ([2**60, 2**60 + 1, 2**60 + 2], [3, 1, 2]),
])
def test_unadjusted_matches_scipy_average_rank_spearman(x, y):
    record = spearman_record(x, y)
    assert record["value"] == pytest.approx(spearmanr(x, y).statistic, abs=1e-14)
    assert record["reason"] is None
    assert record["n"] == len(x)
    assert -1.0 <= record["value"] <= 1.0


def test_strict_monotone_transformations_preserve_ties_and_coefficients():
    x = np.array([1, 2, 2, 5, 3, 6, 4, 4], dtype=float)
    y = np.array([4, 2, 2, 5, 6, 1, 4, 3], dtype=float)
    control = np.array([2, 3, 1, 1, 4, 7, 8, 5], dtype=float)[:, None]
    for controls in (None, control):
        original = spearman_record(x, y, controls)
        transformed_controls = None if controls is None else controls**3
        transformed = spearman_record(np.exp(x), y**3, transformed_controls)
        decreasing = spearman_record(-x, y, controls)
        assert transformed["value"] == pytest.approx(original["value"], abs=1e-14)
        assert decreasing["value"] == pytest.approx(-original["value"], abs=1e-14)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_observations_are_explicitly_undefined(n):
    record = spearman_record(np.arange(n), np.arange(n)[::-1])
    assert record == {"value": None, "reason": "fewer_than_three_particles", "n": n}
    partial = spearman_record(np.arange(n), np.arange(n), np.empty((n, 0)))
    assert partial["value"] is None
    assert partial["reason"] == "fewer_than_three_particles"
    assert partial["control_design_rank"] == min(n, 1)
    assert partial["residual_degrees_of_freedom"] == n - min(n, 1)


@pytest.mark.parametrize("x,y,reason", [
    ([2, 2, 2, 2, 2], [1, 2, 3, 4, 5], "constant_x"),
    ([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], "constant_y"),
])
def test_original_constant_ranks_remain_undefined(x, y, reason):
    for controls in (None, np.zeros((5, 2))):
        record = spearman_record(x, y, controls)
        assert record["value"] is None
        assert record["reason"] == reason
        if controls is not None:
            assert record["control_design_rank"] == 1
            assert record["residual_degrees_of_freedom"] == 4


@pytest.mark.parametrize("x,y", [
    (1.0, [1, 2, 3]),
    ([1, 2, 3], 1.0),
    ([[1, 2, 3]], [1, 2, 3]),
    ([1, 2, 3], [[1], [2], [3]]),
    ([1, 2], [1, 2, 3]),
    ([1, 2, np.nan], [1, 2, 3]),
    ([1, 2, 3], [1, np.inf, 3]),
    ([1, 2, -np.inf], [1, 2, 3]),
    ([1, 2, 3j], [1, 2, 3]),
    (["1", "2", "3"], [1, 2, 3]),
    ([False, True, True], [1, 2, 3]),
    (np.array([1, 2, 3], dtype=object), [1, 2, 3]),
    ([[1, 2], [3]], [1, 2]),
])
def test_malformed_or_nonfinite_vectors_raise_without_dropping_rows(x, y):
    with pytest.raises(ValueError):
        spearman_record(x, y)


@pytest.mark.parametrize("controls", [
    [1, 2, 3, 4, 5],
    np.ones((4, 2)),
    np.ones((5, 1, 1)),
    [[1], [2], [3], [4], [np.nan]],
    [[1], [2], [3], [4], [np.inf]],
    np.ones((5, 1), dtype=complex),
    np.full((5, 1), "1"),
    np.ones((5, 1), dtype=bool),
])
def test_malformed_controls_raise_even_when_the_response_is_constant(controls):
    with pytest.raises(ValueError):
        spearman_record(np.ones(5), np.arange(5), controls)


@pytest.mark.parametrize("controlled_response", ["x", "y"])
def test_perfect_rank_control_does_not_turn_roundoff_into_a_relationship(controlled_response):
    controlled = np.array([4, 2, 6, 3, 7, 8, 1, 5])
    other = np.array([8, 2, 5, 6, 4, 3, 7, 1])
    x, y = (controlled, other) if controlled_response == "x" else (other, controlled)
    record = spearman_record(x, y, controlled[:, None] ** 3)
    assert record["value"] is None
    assert record["reason"] == f"negligible_{controlled_response}_residual"
    assert record["control_design_rank"] == 2
    assert record["residual_degrees_of_freedom"] == 6


def test_rank_deficient_controls_preserve_their_column_space_adjustment():
    x = np.array([1, 5, 2, 7, 4, 8, 3, 6])
    y = np.array([7, 2, 3, 6, 5, 8, 1, 4])
    z = np.array([2, 1, 6, 8, 7, 3, 4, 5])
    single = spearman_record(x, y, z[:, None])
    redundant = spearman_record(x, y, np.column_stack((z, z**3, np.ones(8))))
    assert single["value"] is not None
    assert redundant["value"] == pytest.approx(single["value"], abs=1e-14)
    assert redundant["control_design_rank"] == 2
    assert redundant["residual_degrees_of_freedom"] == 6


def test_constant_or_empty_controls_fit_only_an_intercept():
    x, y = [2, 1, 5, 3, 4], [5, 1, 3, 4, 2]
    expected = spearman_record(x, y)["value"]
    for controls in (np.empty((5, 0)), np.zeros((5, 2))):
        record = spearman_record(x, y, controls)
        assert record["value"] == pytest.approx(expected, abs=1e-14)
        assert record["control_design_rank"] == 1
        assert record["residual_degrees_of_freedom"] == 4


def test_insufficient_residual_degrees_of_freedom_are_not_a_coefficient():
    record = spearman_record([4, 1, 3, 2], [2, 4, 1, 3], np.arange(4)[:, None])
    assert record["value"] is None
    assert record["reason"] == "insufficient_residual_degrees_of_freedom"
    assert record["control_design_rank"] == 2
    assert record["residual_degrees_of_freedom"] == 2
    intercept_only = spearman_record([1, 2, 3], [1, 3, 2], np.empty((3, 0)))
    assert intercept_only["reason"] == "insufficient_residual_degrees_of_freedom"


def test_one_control_matches_the_independent_partial_correlation_formula():
    x = [2, 4, 4, 1, 6, 8, 7, 3, 5]
    y = [1, 3, 2, 5, 7, 8, 4, 6, 3]
    z = [3, 2, 2, 5, 6, 8, 7, 1, 4]
    xy = spearmanr(x, y).statistic
    xz = spearmanr(x, z).statistic
    yz = spearmanr(y, z).statistic
    expected = (xy - xz * yz) / np.sqrt((1 - xz**2) * (1 - yz**2))
    record = spearman_record(x, y, np.asarray(z)[:, None])
    assert record["value"] == pytest.approx(expected, abs=1e-14)
    assert record["control_design_rank"] == 2
    assert record["residual_degrees_of_freedom"] == 7


def test_multiple_controls_match_an_independent_qr_projection():
    x = np.array([2, 6, 3, 8, 3, 1, 7, 5, 4, 9])
    y = np.array([5, 1, 9, 4, 7, 6, 8, 3, 2, 5])
    controls = np.column_stack(([1, 4, 7, 2, 9, 3, 10, 8, 5, 6],
                                [4, 3, 9, 8, 1, 10, 2, 7, 6, 5]))
    ranked_controls = rankdata(controls, method="average", axis=0)
    orthonormal, _ = np.linalg.qr(np.column_stack((np.ones(10), ranked_controls)))
    rx, ry = rankdata(x), rankdata(y)
    residual_x = rx - orthonormal @ (orthonormal.T @ rx)
    residual_y = ry - orthonormal @ (orthonormal.T @ ry)
    expected = np.corrcoef(residual_x, residual_y)[0, 1]
    originals = x.copy(), y.copy(), controls.copy()
    record = spearman_record(x, y, controls)
    assert record["value"] == pytest.approx(expected, abs=1e-14)
    assert record["control_design_rank"] == 3
    assert record["residual_degrees_of_freedom"] == 7
    assert record["reason"] is None
    for current, original in zip((x, y, controls), originals):
        np.testing.assert_array_equal(current, original)
