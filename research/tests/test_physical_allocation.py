import numpy as np
import pytest

from research.physical_allocation import physical_scores


def example():
    velocities = np.array([[0., 0.], [3., 4.], [0., 4.], [0., 0.], [8., 15.]])
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2]], dtype=np.int64)
    return velocities, pairs


def test_analytic_irregular_degrees_velocity_differences_and_isolate():
    velocities, pairs = example()
    result = physical_scores(velocities, pairs)
    np.testing.assert_array_equal(result["neighbor_count"], [3, 2, 2, 1, 0])
    np.testing.assert_array_equal(result["negative_neighbor_count"], [-3., -2., -2., -1., 0.])
    np.testing.assert_allclose(result["velocity_dispersion"],
                               [np.sqrt(41. / 3.), np.sqrt(17.), np.sqrt(12.5), 0., 0.])
    np.testing.assert_array_equal(result["isolated"], [False, False, False, False, True])
    assert result["negative_neighbor_count"].dtype == np.float64
    assert result["velocity_dispersion"].dtype == np.float64
    assert result["neighbor_count"].dtype == np.int64
    assert result["isolated"].dtype == np.bool_


def test_regular_degree_cycle_and_uniform_translation_have_zero_dispersion():
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    result = physical_scores(np.tile([3., 4.], (4, 1)), pairs)
    np.testing.assert_array_equal(result["neighbor_count"], 2)
    np.testing.assert_array_equal(result["velocity_dispersion"], 0.)
    assert not result["isolated"].any()


def test_empty_graph_and_single_particle_make_explicit_zero_isolate_scores():
    result = physical_scores([[8., -5.]], np.empty((0, 2), dtype=int))
    for key in ("negative_neighbor_count", "velocity_dispersion", "neighbor_count"):
        np.testing.assert_array_equal(result[key], [0])
    np.testing.assert_array_equal(result["isolated"], [True])


def test_galilean_velocity_offset_rotation_and_reflection_preserve_scores():
    velocities, pairs = example()
    original = physical_scores(velocities, pairs)
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for transformed in (velocities + [9., -7.], velocities @ rotation.T,
                        velocities @ np.diag([-1., 1.])):
        actual = physical_scores(transformed, pairs)
        for key in original:
            np.testing.assert_allclose(actual[key], original[key], atol=1e-12)


def test_particle_permutation_is_equivariant_and_edge_order_unrestricted():
    velocities, pairs = example()
    original = physical_scores(velocities, pairs)
    permutation = np.array([4, 1, 3, 0, 2])
    inverse = np.argsort(permutation)
    transformed_pairs = np.sort(inverse[pairs], axis=1)[::-1]
    actual = physical_scores(velocities[permutation], transformed_pairs)
    for key in original:
        np.testing.assert_allclose(actual[key], original[key][permutation], atol=1e-12)


def test_supplied_graph_is_not_recomputed_and_inputs_are_unchanged():
    velocities, pairs = example()
    velocity_copy, pair_copy = velocities.copy(), pairs.copy()
    result = physical_scores(velocities, pairs)
    fewer = physical_scores(velocities, pairs[1:])
    np.testing.assert_array_equal(fewer["neighbor_count"], result["neighbor_count"] - [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(velocities, velocity_copy)
    np.testing.assert_array_equal(pairs, pair_copy)


def test_stable_rms_avoids_unnecessary_square_overflow():
    velocities = np.array([[0., 0.], [3e200, 4e200]])
    result = physical_scores(velocities, np.array([[0, 1]]))
    np.testing.assert_allclose(result["velocity_dispersion"], [5e200, 5e200])


@pytest.mark.parametrize("bad", [[], [[0., 0., 0.]], [0., 0.], [[np.nan, 0.]],
    [[np.inf, 0.]], [[1j, 0.]], [["1", "2"]], [[True, False]]])
def test_malformed_velocities_are_rejected(bad):
    with pytest.raises(ValueError):
        physical_scores(bad, np.empty((0, 2), dtype=int))


@pytest.mark.parametrize("bad", [
    np.array([0, 1]), np.zeros((1, 3), dtype=int), np.array([[0., 1.]]),
    np.array([[True, False]]), np.array([[0, 0]]), np.array([[1, 0]]),
    np.array([[-1, 0]]), np.array([[0, 2]]), np.array([[0, 1], [0, 1]]),
    np.array([[0., np.nan]]), np.array([[0., np.inf]]), np.array([[0j, 1j]]),
    np.array([["0", "1"]]), np.array([[0, 2**64 - 1]], dtype=np.uint64),
])
def test_malformed_pairs_are_rejected(bad):
    with pytest.raises(ValueError):
        physical_scores([[0., 0.], [1., 1.]], bad)


def test_unsigned_valid_pairs_and_integer_velocities_are_supported():
    result = physical_scores([[0, 0], [3, 4]], np.array([[0, 1]], dtype=np.uint64))
    np.testing.assert_allclose(result["velocity_dispersion"], [5., 5.])


def test_unrepresentable_arithmetic_raises_instead_of_imputing_scores():
    with pytest.raises(ValueError, match="arithmetic"):
        physical_scores([[1e308, 0.], [-1e308, 0.]], np.array([[0, 1]]))
