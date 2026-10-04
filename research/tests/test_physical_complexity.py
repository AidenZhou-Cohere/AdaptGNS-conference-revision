import numpy as np
import pytest

from research.physical_complexity import local_flow_diagnostics


def grid():
    return np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])


def test_uniform_translation_has_zero_local_derivatives_and_dispersion():
    positions = grid()
    velocities = np.tile([3., 4.], (len(positions), 1))
    result = local_flow_diagnostics(positions, velocities, radius=3.)
    np.testing.assert_array_equal(result["neighbor_count"], 8)
    np.testing.assert_array_equal(result["fit_valid"], True)
    np.testing.assert_allclose(result["speed"], 5.)
    for key in ("gradient", "strain_frobenius", "abs_divergence",
                "abs_vorticity", "velocity_dispersion"):
        np.testing.assert_array_equal(result[key], 0.)


@pytest.mark.parametrize("gradient, strain, vorticity, divergence", [
    (np.array([[0., 2.], [0., 0.]]), np.sqrt(2.), 2., 0.),
    (np.array([[0., -2.], [2., 0.]]), 0., 4., 0.),
    (np.array([[2., 0.], [0., 2.]]), np.sqrt(8.), 0., 4.),
    (np.array([[-3., 1.], [2., 4.]]), np.sqrt(29.5), 1., 1.),
])
def test_affine_shear_rotation_expansion_and_general_flow(gradient, strain, vorticity, divergence):
    positions = grid()
    velocities = positions @ gradient.T + [5., -8.]
    result = local_flow_diagnostics(positions, velocities, radius=3.)
    np.testing.assert_allclose(result["gradient"], np.broadcast_to(gradient, (9, 2, 2)), atol=1e-12)
    np.testing.assert_allclose(result["strain_frobenius"], strain, atol=1e-12)
    np.testing.assert_allclose(result["abs_vorticity"], vorticity, atol=1e-12)
    np.testing.assert_allclose(result["abs_divergence"], divergence, atol=1e-12)
    # This compares dispersion to its definition independently of the fit.
    expected = [np.sqrt(np.mean(np.sum((np.delete(velocities, i, axis=0) - velocities[i]) ** 2, axis=1)))
                 for i in range(len(positions))]
    np.testing.assert_allclose(result["velocity_dispersion"], expected, atol=1e-12)


def test_rigid_coordinate_rotation_and_translation_preserve_scalar_proxies():
    positions = grid()
    gradient = np.array([[1., -4.], [2., -3.]])
    velocities = positions @ gradient.T + [2., 3.]
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original = local_flow_diagnostics(positions, velocities, radius=3.)
    transformed = local_flow_diagnostics(positions @ rotation.T + [12., 5.],
                                       velocities @ rotation.T, radius=3.)
    for key in original:
        if key == "gradient":
            np.testing.assert_allclose(transformed[key],
                rotation @ original[key] @ rotation.T, atol=1e-12)
        else:
            np.testing.assert_allclose(transformed[key], original[key], atol=1e-12)


def test_velocity_offset_leaves_local_proxies_unchanged_but_speed_can_change():
    positions = grid()
    velocities = positions @ np.array([[1., 2.], [-3., 4.]])
    original = local_flow_diagnostics(positions, velocities, radius=3.)
    shifted = local_flow_diagnostics(positions, velocities + [8., -11.], radius=3.)
    for key in original.keys() - {"speed"}:
        np.testing.assert_allclose(shifted[key], original[key], atol=1e-12)
    assert not np.allclose(shifted["speed"], original["speed"])


def test_strict_radius_float64_boundary_self_exclusion_and_duplicate_ids():
    radius = 0.015
    positions = np.array([[0., 0.], [np.nextafter(radius, 0.), 0.],
                          [radius, 0.], [np.nextafter(radius, np.inf), 0.],
                          [0., 0.]])
    result = local_flow_diagnostics(positions, np.zeros_like(positions))
    # At the origin, only the strictly interior point and the distinct particle
    # at the same coordinates are neighbors. The particle itself is not.
    assert result["neighbor_count"][0] == 2
    assert result["neighbor_count"][4] == 2
    assert not result["fit_valid"].any()
    assert np.isinf(result["fit_condition"]).all()


def test_tree_candidate_box_does_not_include_outside_euclidean_radius():
    positions = np.array([[0., 0.], [0.8, 0.8], [0.4, 0.4]])
    result = local_flow_diagnostics(positions, np.zeros_like(positions), radius=1.)
    assert result["neighbor_count"][0] == 1


def test_sparse_and_empty_neighborhoods_preserve_undefined_values():
    positions = np.array([[0., 0.], [1., 0.], [0., 1.], [20., 20.]])
    result = local_flow_diagnostics(positions, positions.copy(), radius=1.1)
    np.testing.assert_array_equal(result["neighbor_count"], [2, 1, 1, 0])
    assert not result["fit_valid"].any()
    assert result["fit_condition"][0] == pytest.approx(1.)
    assert np.isinf(result["fit_condition"][[1, 2]]).all()
    assert np.isnan(result["fit_condition"][3])
    assert np.isfinite(result["velocity_dispersion"][:3]).all()
    assert np.isnan(result["velocity_dispersion"][3])
    assert np.isfinite(result["speed"]).all()
    for key in ("gradient", "strain_frobenius", "abs_vorticity", "abs_divergence"):
        assert np.isnan(result[key]).all()
    # Lowering the count threshold does not waive the independent rank check.
    permissive = local_flow_diagnostics(positions, positions.copy(), radius=1.1, min_neighbors=1)
    np.testing.assert_array_equal(permissive["fit_valid"], [True, False, False, False])
    np.testing.assert_allclose(permissive["gradient"][0], np.eye(2), atol=1e-12)


def test_collinear_neighbors_have_no_unique_two_dimensional_gradient():
    positions = np.array([[x, 0.] for x in range(5)], dtype=float)
    result = local_flow_diagnostics(positions, positions.copy(), radius=10.)
    np.testing.assert_array_equal(result["neighbor_count"], 4)
    assert not result["fit_valid"].any()
    assert np.isinf(result["fit_condition"]).all()
    assert np.isnan(result["gradient"]).all()


def test_gram_condition_cutoff_is_inclusive_and_is_squared_design_condition():
    positions = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., .25], [0., -.25]])
    result = local_flow_diagnostics(positions, positions.copy(), radius=2., max_condition=16.)
    assert result["fit_condition"][0] == pytest.approx(16.)
    assert result["fit_valid"][0]
    rejected = local_flow_diagnostics(positions, positions.copy(), radius=2., max_condition=15.9)
    assert not rejected["fit_valid"][0]
    assert np.isnan(rejected["gradient"][0]).all()


def test_particle_permutation_and_input_storage_do_not_change_outputs():
    positions = grid()
    velocities = positions @ np.array([[1., 2.], [-3., 4.]])
    position_copy, velocity_copy = positions.copy(), velocities.copy()
    expected = local_flow_diagnostics(positions, velocities, radius=3.)
    permutation = np.array([8, 1, 5, 0, 3, 6, 2, 4, 7])
    actual = local_flow_diagnostics(positions[permutation], velocities[permutation], radius=3.)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key][permutation], atol=1e-12)
    np.testing.assert_array_equal(positions, position_copy)
    np.testing.assert_array_equal(velocities, velocity_copy)


@pytest.mark.parametrize("bad", [[], [[0., 0., 0.]], [0., 0.],
    [[np.nan, 0.]], [[np.inf, 0.]], [[1j, 0.]], [["1", "2"]], [[True, False]]])
@pytest.mark.parametrize("argument", ["positions", "velocities"])
def test_malformed_states_are_rejected(bad, argument):
    arguments = {"positions": [[0., 0.]], "velocities": [[0., 0.]]}
    arguments[argument] = bad
    with pytest.raises(ValueError):
        local_flow_diagnostics(**arguments)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        local_flow_diagnostics([[0., 0.]], [[0., 0.], [1., 1.]])


@pytest.mark.parametrize("kwargs", [
    {"radius": 0.}, {"radius": -1.}, {"radius": np.inf}, {"radius": np.nan},
    {"radius": True}, {"radius": "0.1"}, {"radius": 1e-200},
    {"radius": 1e200}, {"min_neighbors": 0}, {"min_neighbors": -1},
    {"min_neighbors": 3.}, {"min_neighbors": True}, {"max_condition": .5},
    {"max_condition": np.inf}, {"max_condition": np.nan}, {"max_condition": True},
])
def test_malformed_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        local_flow_diagnostics([[0., 0.]], [[0., 0.]], **kwargs)


def test_numeric_numpy_scalar_parameters_and_single_particle_are_supported():
    result = local_flow_diagnostics([[0, 0]], [[3, 4]], radius=np.float64(.015),
                                   min_neighbors=np.int64(3), max_condition=np.float64(1e6))
    assert result["speed"][0] == 5.
    assert result["neighbor_count"][0] == 0
    assert not result["fit_valid"][0]
    assert np.isnan(result["velocity_dispersion"][0])


def test_unrepresentable_input_arithmetic_is_not_silently_imputed():
    with pytest.raises(ValueError, match="arithmetic"):
        local_flow_diagnostics([[0., 0.], [.001, 0.]],
                               [[1e308, 0.], [-1e308, 0.]])
