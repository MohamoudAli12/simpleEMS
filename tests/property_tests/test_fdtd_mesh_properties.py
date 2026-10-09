"""Property tests for :mod:`simpleEMS.fdtd_mesh`.

The auto mesher stands on a few small helpers: one collapses near-duplicate
line positions without losing a must-keep line, one thins a curve's facet
vertices down to the mesh resolution, and one grades the lines across an
interval as a geometric series. A broken invariant in any of them shows up
much later as a runaway timestep or a lost port line, so they are checked
here directly, over drawn inputs, instead of through a full geometry build.
"""

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# simpleEMS imports CSXCAD at module scope, so without it this module cannot
# even be collected.
pytest.importorskip("CSXCAD")

from simpleEMS.fdtd_mesh import (  # noqa: E402
    _decimate_curve_coords,
    _factor_for_num,
    _geom_dist,
    _lines_const_factor_in_bounds,
    _remove_dups,
    fp_equalp,
)

# Positions packed into 1e-3 mm, so rounding to the mesher's 5 decimal
# places makes plenty of them collide
crowded_positions = st.floats(min_value=0.0, max_value=1e-3)
coordinates = st.floats(min_value=-50.0, max_value=50.0)
min_features = st.floats(min_value=1e-3, max_value=5.0)
spacings = st.floats(min_value=0.05, max_value=2.0)
growth_factors = st.floats(min_value=1.0, max_value=2.0)
smoothness = st.floats(min_value=1.2, max_value=2.0)


@st.composite
def sorted_lines_with_fixed(draw):
    """Sorted line positions plus a subset of them that must survive."""
    lines = sorted(draw(st.lists(crowded_positions, min_size=1, max_size=40)))
    fixed = draw(st.lists(st.sampled_from(lines), max_size=5, unique=True))
    # two must-keep lines that round together cannot both survive
    assume(
        all(
            not fp_equalp(first, second)
            for index, first in enumerate(fixed)
            for second in fixed[index + 1 :]
        )
    )
    return lines, fixed


class TestRemoveDups:
    @given(sorted_lines_with_fixed())
    def test_no_two_neighbours_round_to_the_same_position(self, lines_and_fixed):
        lines, fixed = lines_and_fixed

        kept = _remove_dups(lines, fixed)

        assert all(not fp_equalp(lower, upper) for lower, upper in pairwise(kept))

    @given(sorted_lines_with_fixed())
    def test_every_fixed_line_survives(self, lines_and_fixed):
        lines, fixed = lines_and_fixed

        kept = _remove_dups(lines, fixed)

        assert set(fixed) <= set(kept)

    @given(sorted_lines_with_fixed())
    def test_the_result_is_a_sorted_subset_of_the_input(self, lines_and_fixed):
        lines, fixed = lines_and_fixed

        kept = _remove_dups(lines, fixed)

        assert kept == sorted(kept)
        assert set(kept) <= set(lines)

    @given(sorted_lines_with_fixed())
    def test_every_input_line_has_a_kept_line_at_its_rounded_position(
        self, lines_and_fixed
    ):
        lines, fixed = lines_and_fixed

        kept = _remove_dups(lines, fixed)

        assert all(any(fp_equalp(line, other) for other in kept) for line in lines)


class TestDecimateCurveCoords:
    @given(st.lists(coordinates, min_size=1, max_size=60), min_features)
    def test_keeps_the_full_extent(self, coords, min_feature):
        kept = _decimate_curve_coords(coords, min_feature)

        assert kept[0] == min(coords)
        assert fp_equalp(kept[-1], max(coords))

    @given(st.lists(coordinates, min_size=1, max_size=60), min_features)
    def test_kept_values_are_sorted_input_values(self, coords, min_feature):
        kept = _decimate_curve_coords(coords, min_feature)

        assert kept == sorted(kept)
        assert set(kept) <= set(coords)

    @given(st.lists(coordinates, min_size=1, max_size=60), min_features)
    def test_kept_values_are_at_least_a_feature_apart(self, coords, min_feature):
        kept = _decimate_curve_coords(coords, min_feature)
        gaps = np.diff(kept)

        # a shape narrower than one feature still keeps both of its edges
        if len(kept) == 2:
            return
        assert np.all(gaps >= min_feature)


class TestGeometricSeries:
    @given(
        factor=growth_factors,
        num=st.integers(min_value=2, max_value=30),
        first_spacing=spacings,
    )
    def test_factor_for_num_inverts_the_series_length(self, factor, num, first_spacing):
        dist = _geom_dist(factor, num, first_spacing)

        recovered = _factor_for_num(num, first_spacing, dist)

        assert _geom_dist(recovered, num, first_spacing) == pytest.approx(
            dist, rel=1e-6
        )

    # an open-ended solve from 1.5 used to stall on these, and with the factor
    # stuck there _geom_series never left its loop
    @pytest.mark.parametrize(
        ("factor", "num", "first_spacing"),
        [(2.0, 14, 1.0), (1.002, 2000, 0.05)],
        ids=["steep", "long"],
    )
    def test_factor_for_num_inverts_extreme_series(self, factor, num, first_spacing):
        dist = _geom_dist(factor, num, first_spacing)

        recovered = _factor_for_num(num, first_spacing, dist)

        assert _geom_dist(recovered, num, first_spacing) == pytest.approx(
            dist, rel=1e-6
        )

    @given(
        lower_spacing=spacings,
        upper_spacing=spacings,
        dist=st.floats(min_value=1.0, max_value=100.0),
        min_lines=st.integers(min_value=2, max_value=5),
        smooth=smoothness,
    )
    def test_graded_lines_span_the_interval_without_breaking_smoothness(
        self, lower_spacing, upper_spacing, dist, min_lines, smooth
    ):
        lines = _lines_const_factor_in_bounds(
            0.0, dist, lower_spacing, upper_spacing, 0, min_lines, smooth
        )
        cells = np.diff(lines)
        # the last cell is trimmed to land exactly on the interval end
        ratios = cells[1:-1] / cells[:-2]

        assert fp_equalp(lines[0], 0.0)
        assert lines[-1] == dist
        assert len(lines) >= min_lines
        assert np.all(cells > 0)
        assert np.all(ratios <= smooth * (1 + 1e-6))
        assert np.all(1 / ratios <= smooth * (1 + 1e-6))

    def test_a_crowded_interval_is_spaced_evenly(self):
        """min_lines asks for more lines than fit at either target spacing;
        a shrinking series used to steepen past ``smooth`` here."""
        lines = _lines_const_factor_in_bounds(0.0, 1.0, 0.75, 1.0, 0, 4, 1.5)

        np.testing.assert_allclose(lines, np.linspace(0.0, 1.0, 4))
