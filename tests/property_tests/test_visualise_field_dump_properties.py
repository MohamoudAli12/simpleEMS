"""Property tests for :mod:`simpleEMS.visualise_field_dump`.

The viewer paints the field onto the structure and resamples it for volume
plots through one sparse matrix of trilinear weights. These properties pin what
any interpolation of grid data must do, for any graded mesh of the kind the
FDTD mesher builds.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from simpleEMS.visualise_field_dump import _interpolation_matrix

# Graded mesh lines: 2 to 12 lines per axis, spacings from 1 um to 5 mm, the
# range from a copper layer to a coarse air region.
spacings_mm = st.lists(
    st.floats(min_value=1e-3, max_value=5.0), min_size=1, max_size=11
)
coefficients = st.floats(min_value=-1e3, max_value=1e3)
fractions = st.lists(
    st.tuples(*[st.floats(min_value=0.0, max_value=1.0)] * 3), min_size=1, max_size=20
)


def lines_from(spacings):
    return np.concatenate([[0.0], np.cumsum(spacings)])


def points_inside(lines, point_fractions):
    """Map fractions in [0, 1] to points inside the grid."""
    low = np.array([axis_lines[0] for axis_lines in lines])
    high = np.array([axis_lines[-1] for axis_lines in lines])
    return low + np.array(point_fractions) * (high - low)


@given(spacings_mm, spacings_mm, spacings_mm, fractions)
def test_the_weights_of_an_inside_point_sum_to_one(
    x_spacings, y_spacings, z_spacings, point_fractions
):
    lines = [lines_from(spacings) for spacings in (x_spacings, y_spacings, z_spacings)]
    points = points_inside(lines, point_fractions)

    weights = _interpolation_matrix(*lines, points)

    np.testing.assert_allclose(weights.sum(axis=1).A1, 1.0, atol=1e-9)
    assert weights.min() >= 0.0


@given(
    spacings_mm,
    spacings_mm,
    spacings_mm,
    fractions,
    coefficients,
    coefficients,
    coefficients,
)
def test_reproduces_any_linear_field(
    x_spacings, y_spacings, z_spacings, point_fractions, slope_x, slope_y, slope_z
):
    lines = [lines_from(spacings) for spacings in (x_spacings, y_spacings, z_spacings)]
    points = points_inside(lines, point_fractions)
    z_mesh, y_mesh, x_mesh = np.meshgrid(lines[2], lines[1], lines[0], indexing="ij")
    grid_values = slope_x * x_mesh + slope_y * y_mesh + slope_z * z_mesh

    weights = _interpolation_matrix(*lines, points)

    expected = points @ np.array([slope_x, slope_y, slope_z])
    scale = 1.0 + np.abs(grid_values).max()
    np.testing.assert_allclose(
        weights @ grid_values.ravel(), expected, atol=1e-9 * scale
    )
