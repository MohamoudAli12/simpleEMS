"""Unit tests for the pure helpers in :mod:`simpleEMS.fdtd_mesh`.

``fdtd_mesh`` is the largest untested module in the package and the one whose
output is hardest to eyeball: a subtly wrong grid produces a simulation that
runs to completion and returns plausible-looking but wrong S-parameters. The
geometric-series machinery below is pure numerics with checkable invariants
(spans the interval, growth factor stays under the smoothing cap, endpoints
land exactly), so it is tested directly rather than through a built structure.

The ``Mesh`` class itself needs real CSXCAD primitives and is covered in
``test_geometry.py``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS import fdtd_mesh  # noqa: E402
from simpleEMS.fdtd_mesh import (  # noqa: E402
    PREC,
    BoundedType,
    Type,
    _dist_for_max_spacings,
    _factor_for_num,
    _factor_ubound,
    _float_inside,
    _geom_dist,
    _geom_dist_zero,
    _geom_series,
    _lines_const_factor_in_bounds,
    _num_for_factor,
    _pos_in_bounds,
    _remove_dups,
    _sort_bounded_types,
    _spacing_at_dist,
    fp_equalp,
    fp_gep,
    fp_gtp,
    fp_lep,
    fp_ltp,
    fp_nearest,
)


# ---------------------------------------------------------------------
# Floating-point comparators
# ---------------------------------------------------------------------
class TestFloatingPointComparators:
    def test_fp_nearest_rounds_to_prec(self):
        assert fp_nearest(1.0 + 1e-25) == 1.0
        assert fp_nearest(1e-21) == 0.0

    def test_fp_nearest_accepts_arrays(self):
        result = fp_nearest(np.array([1e-21, 2.5, 3.5]))

        assert result == pytest.approx([0.0, 2.5, 3.5])

    def test_tolerance_is_absolute_not_relative(self):
        """Rounding to 5 decimals means the tolerance vanishes for
        millimetre-scale coordinates -- there, these are exact comparisons.
        Only sub-1e-5 values are actually collapsed."""
        assert fp_equalp(1e-7, 2e-7) is np.True_
        assert not fp_equalp(1.0, 1.00001)

    @pytest.mark.parametrize(
        ("func", "a", "b", "expected"),
        [
            (fp_equalp, 1.5, 1.5, True),
            (fp_equalp, 1.5, 1.6, False),
            (fp_gtp, 2.0, 1.0, True),
            (fp_gtp, 1.0, 1.0, False),
            (fp_gtp, 1.0, 2.0, False),
            (fp_gep, 1.0, 1.0, True),
            (fp_gep, 2.0, 1.0, True),
            (fp_gep, 0.0, 1.0, False),
            (fp_ltp, 1.0, 2.0, True),
            (fp_ltp, 1.0, 1.0, False),
            (fp_lep, 1.0, 1.0, True),
            (fp_lep, 0.5, 1.0, True),
            (fp_lep, 2.0, 1.0, False),
        ],
    )
    def test_comparator_truth_table(self, func, a, b, expected):
        assert bool(func(a, b)) is expected

    def test_strict_and_non_strict_agree_away_from_equality(self):
        assert bool(fp_gtp(3.0, 1.0)) == bool(fp_gep(3.0, 1.0))
        assert bool(fp_ltp(1.0, 3.0)) == bool(fp_lep(1.0, 3.0))

    def test_comparators_are_mutually_consistent(self):
        for a, b in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (-3.0, 0.0)]:
            assert bool(fp_gtp(a, b)) == bool(fp_ltp(b, a))
            assert bool(fp_gep(a, b)) == bool(fp_lep(b, a))
            assert bool(fp_gep(a, b)) == (bool(fp_gtp(a, b)) or bool(fp_equalp(a, b)))

    def test_prec_absorbs_last_place_noise(self):
        """``PREC`` has to collapse the ~1e-16 discrepancy between an edge
        written as a literal and the same edge reached through trigonometry,
        or the mesher builds a degenerate interval between them. It must stay
        well clear of real geometry, so it is set as tight as that allows."""
        assert PREC == 5


# ---------------------------------------------------------------------
# _remove_dups
# ---------------------------------------------------------------------
class TestRemoveDups:
    def test_collapses_exact_duplicates(self):
        assert _remove_dups([1.0, 1.0, 2.0, 3.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_preserves_order_and_distinct_values(self):
        assert _remove_dups([-2.0, 0.0, 1.5, 4.0]) == [-2.0, 0.0, 1.5, 4.0]

    def test_empty_input(self):
        assert _remove_dups([]) == []

    def test_single_value(self):
        assert _remove_dups([1.0]) == [1.0]

    def test_all_identical_collapses_to_one(self):
        assert _remove_dups([2.0] * 5) == [2.0]

    def test_fixed_value_survives_against_a_near_duplicate(self):
        """A must-keep position (an explicit mesh line) wins over the
        neighbour it collides with; dropping it instead would move a mesh line
        off a metal edge."""
        assert _remove_dups([1e-21, 2e-21, 5.0], fixed=[2e-21]) == [2e-21, 5.0]

    def test_without_fixed_the_earlier_value_survives(self):
        assert _remove_dups([1e-21, 2e-21, 5.0]) == [1e-21, 5.0]

    def test_none_fixed_is_treated_as_empty(self):
        assert _remove_dups([1.0, 1.0, 2.0], fixed=None) == _remove_dups(
            [1.0, 1.0, 2.0], fixed=[]
        )

    def test_output_never_grows(self):
        values = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0]

        assert len(_remove_dups(values)) <= len(values)


# ---------------------------------------------------------------------
# BoundedType
# ---------------------------------------------------------------------
class TestBoundedType:
    def test_accessors_round_trip_the_constructor(self):
        interval = BoundedType(Type.metal, 1.0, 3.0)

        assert interval.get_type() is Type.metal
        assert interval.get_bounds() == [1.0, 3.0]

    def test_size_is_the_interval_length(self):
        assert BoundedType(Type.air, -2.0, 3.0).size() == 5.0

    def test_midpoint_is_the_average_of_the_bounds(self):
        assert BoundedType(Type.nonmetal, -2.0, 4.0).get_midpoint() == 1.0

    def test_zero_length_interval(self):
        """Fixed lines are emitted as zero-length intervals."""
        interval = BoundedType(Type.metal, 2.5, 2.5)

        assert interval.size() == 0.0
        assert interval.get_midpoint() == 2.5

    def test_type_enum_members(self):
        assert {member.name for member in Type} == {"metal", "nonmetal", "air"}

    def test_sorting_orders_by_increasing_size(self):
        """Meshing processes the finest features first, so ordering is
        behaviour, not presentation."""
        intervals = [
            BoundedType(Type.air, 0.0, 10.0),
            BoundedType(Type.metal, 0.0, 0.5),
            BoundedType(Type.nonmetal, 0.0, 3.0),
        ]

        ordered = _sort_bounded_types([intervals, [], []])

        assert [round(i.size(), 6) for i in ordered[0]] == [0.5, 3.0, 10.0]

    def test_sorting_returns_all_three_dimensions(self):
        ordered = _sort_bounded_types([[], [], []])

        assert len(ordered) == 3


# ---------------------------------------------------------------------
# Position predicates
# ---------------------------------------------------------------------
class TestPositionPredicates:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, True), (5.0, True), (10.0, True), (-0.1, False), (10.1, False)],
    )
    def test_float_inside_is_inclusive(self, value, expected):
        assert _float_inside(value, 0.0, 10.0) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, True), (5.0, True), (10.0, True), (-0.1, False), (10.1, False)],
    )
    def test_pos_in_bounds_is_inclusive(self, value, expected):
        assert bool(_pos_in_bounds(value, 0.0, 10.0)) is expected

    def test_pos_in_bounds_agrees_with_float_inside_at_mm_scale(self):
        """The two differ only inside the fp tolerance, which is nil here."""
        for value in np.linspace(-1, 11, 41):
            assert bool(_pos_in_bounds(value, 0.0, 10.0)) == _float_inside(
                value, 0.0, 10.0
            )


# ---------------------------------------------------------------------
# Geometric series primitives
# ---------------------------------------------------------------------
class TestGeomDist:
    def test_unit_factor_is_a_uniform_run(self):
        """factor 1 -> (num - 1) equal steps."""
        assert _geom_dist(1.0, 5, 2.0) == pytest.approx(8.0)

    def test_matches_the_closed_form_geometric_sum(self):
        factor, num, spacing = 1.3, 6, 0.4
        expected = spacing * sum(factor**p for p in range(1, num))

        assert _geom_dist(factor, num, spacing) == pytest.approx(expected, rel=1e-12)

    def test_distance_grows_with_the_factor(self):
        distances = [_geom_dist(f, 6, 1.0) for f in (1.0, 1.2, 1.5, 2.0)]

        assert distances == sorted(distances)

    def test_single_step_spans_nothing(self):
        assert _geom_dist(1.5, 1, 1.0) == 0.0

    def test_zero_objective_is_dist_offset(self):
        value = _geom_dist(1.3, 6, 0.4)

        assert _geom_dist_zero(1.3, 6, 0.4, value) == pytest.approx(0.0, abs=1e-12)


class TestFactorForNum:
    @pytest.mark.parametrize("dist", [5.0, 10.0, 37.5])
    @pytest.mark.parametrize("num", [4, 6, 10])
    def test_solved_factor_reproduces_the_distance(self, num, dist):
        """The whole point of the solve: plugging the root back in must span
        exactly ``dist``."""
        spacing = 1.0
        factor = _factor_for_num(num, spacing, dist)

        assert _geom_dist(factor, num, spacing) == pytest.approx(dist, rel=1e-6)

    def test_uniform_case_returns_unit_factor(self):
        """dist == spacing * (num - 1) needs no growth at all."""
        factor = _factor_for_num(5, 2.0, 8.0)

        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_more_steps_need_a_smaller_factor(self):
        factors = [_factor_for_num(n, 1.0, 20.0) for n in (5, 8, 12)]

        assert factors == sorted(factors, reverse=True)


class TestFactorUbound:
    def test_capped_at_max_factor(self):
        assert _factor_ubound(5, 8.0, 1.5) == 1.5

    def test_uncapped_case_returns_the_exact_ratio_root(self):
        assert _factor_ubound(5, 1.2, 1.5) == pytest.approx(1.2 ** (1 / 4), rel=1e-12)

    def test_never_exceeds_max_factor(self):
        for ratio in (1.01, 1.5, 4.0, 50.0):
            assert _factor_ubound(6, ratio, 1.5) <= 1.5


class TestNumForFactor:
    @pytest.mark.parametrize("dist", [3.0, 10.0, 50.0])
    def test_returned_pair_spans_the_distance(self, dist):
        factor, num = _num_for_factor(1.5, 1.0, dist)

        assert _geom_dist(factor, num, 1.0) == pytest.approx(dist, rel=1e-6)

    def test_factor_stays_within_the_cap(self):
        for dist in (3.0, 10.0, 50.0, 200.0):
            factor, _num = _num_for_factor(1.5, 1.0, dist)
            assert 1.0 <= factor <= 1.5 + 1e-9

    def test_step_count_grows_with_distance(self):
        counts = [_num_for_factor(1.5, 1.0, d)[1] for d in (5.0, 20.0, 100.0)]

        assert counts == sorted(counts)

    def test_accepts_a_zero_dimensional_array_distance(self):
        """Callers pass numpy scalars; the implementation flattens them."""
        factor, num = _num_for_factor(1.5, 1.0, np.array(10.0))

        assert num > 0
        assert np.isfinite(factor)


class TestGeomSeries:
    def test_spans_the_requested_distance(self):
        factor, num = _geom_series(1.0, 4.0, 20.0, 5, 1.5)

        assert _geom_dist(factor, num, 1.0) == pytest.approx(20.0, rel=1e-6)

    def test_honours_the_minimum_line_count(self):
        _factor, num = _geom_series(0.5, 1.0, 2.0, 12, 1.5)

        assert num >= 12

    def test_factor_stays_under_the_smoothing_cap(self):
        factor, num = _geom_series(0.5, 5.0, 40.0, 5, 1.5)

        assert factor < _factor_ubound(num, 5.0 / 0.5, 1.5)
        assert factor <= 1.5


class TestLinesConstFactorInBounds:
    def test_equal_spacings_give_a_uniform_grid(self):
        lines = _lines_const_factor_in_bounds(0.0, 10.0, 1.0, 1.0, 0, 5, 1.5)

        assert np.diff(lines) == pytest.approx(np.full(len(lines) - 1, 1.0))

    def test_endpoints_are_exact(self):
        """A mesh line has to land exactly on the interval edge; drift here
        moves the grid off a metal boundary."""
        for lower_spacing, upper_spacing in [(1.0, 1.0), (0.5, 3.0), (3.0, 0.5)]:
            lines = _lines_const_factor_in_bounds(
                -4.0, 11.0, lower_spacing, upper_spacing, 0, 5, 1.5
            )
            assert lines[0] == pytest.approx(-4.0, abs=1e-9)
            assert lines[-1] == pytest.approx(11.0, abs=1e-9)

    def test_lines_are_monotonically_increasing(self):
        for lower_spacing, upper_spacing in [(0.5, 3.0), (3.0, 0.5), (1.0, 1.0)]:
            lines = _lines_const_factor_in_bounds(
                0.0, 10.0, lower_spacing, upper_spacing, 0, 5, 1.5
            )
            assert np.all(np.diff(lines) > 0)

    def test_spacing_grows_from_the_finer_end(self):
        lines = _lines_const_factor_in_bounds(0.0, 10.0, 0.5, 3.0, 0, 5, 1.5)
        spacings = np.diff(lines)

        assert np.all(np.diff(spacings) > 0)

    def test_spacing_shrinks_toward_the_finer_end(self):
        """Mirror case: fine at the upper end instead of the lower."""
        lines = _lines_const_factor_in_bounds(0.0, 10.0, 3.0, 0.5, 0, 5, 1.5)
        spacings = np.diff(lines)

        assert np.all(np.diff(spacings) < 0)

    def test_adjacent_cell_ratio_respects_the_smoothing_limit(self):
        """This is the mesh-quality guarantee openEMS needs; violating it
        causes reflections off the grid itself."""
        smooth = 1.5
        lines = _lines_const_factor_in_bounds(0.0, 40.0, 0.2, 5.0, 0, 5, smooth)
        spacings = np.diff(lines)
        ratios = spacings[1:] / spacings[:-1]

        assert np.all(ratios <= smooth + 1e-9)

    def test_honours_the_minimum_line_count(self):
        lines = _lines_const_factor_in_bounds(0.0, 1.0, 1.0, 1.0, 0, 9, 1.5)

        assert len(lines) >= 9

    def test_nearly_equal_spacings_take_the_uniform_branch(self):
        """The branch triggers on rtol=1e-3, so a 0.05% difference must not
        fall into the geometric path."""
        lines = _lines_const_factor_in_bounds(0.0, 10.0, 1.0, 1.0005, 0, 5, 1.5)
        spacings = np.diff(lines)

        assert spacings == pytest.approx(np.full(len(spacings), spacings[0]))


class TestSpacingGrowth:
    def test_spacing_grows_over_distance(self):
        near = _spacing_at_dist(1.0, 2.0, 1.5)
        far = _spacing_at_dist(1.0, 20.0, 1.5)

        assert 1.0 <= near < far

    def test_meeting_point_lies_inside_the_interval(self):
        total = 20.0
        meet = _dist_for_max_spacings(0.5, 2.0, total, 1.5)

        assert 0.0 < meet < total

    def test_meeting_point_balances_the_two_spacings(self):
        """Both series must reach the same cell size where they meet,
        otherwise there is a jump in the middle of the grid."""
        total, max_factor = 20.0, 1.5
        meet = _dist_for_max_spacings(0.5, 2.0, total, max_factor)

        from_lower = _spacing_at_dist(0.5, meet, max_factor)
        from_upper = _spacing_at_dist(2.0, total - meet, max_factor)

        assert from_lower == pytest.approx(from_upper, rel=1e-3)

    def test_symmetric_spacings_meet_in_the_middle(self):
        meet = _dist_for_max_spacings(1.0, 1.0, 20.0, 1.5)

        assert meet == pytest.approx(10.0, rel=1e-2)

    def test_finer_end_pushes_the_meeting_point_toward_itself(self):
        """The end that starts finer needs more distance to catch up, so the
        meeting point sits away from the coarser end."""
        meet = _dist_for_max_spacings(0.2, 4.0, 20.0, 1.5)

        assert meet > 10.0


# ---------------------------------------------------------------------
# Polygon and polyhedron helpers
# ---------------------------------------------------------------------


def _metal_linpoly(points, norm_dir=2, elevation=0.0, length=0.035):
    from CSXCAD import ContinuousStructure

    csx = ContinuousStructure()
    metal = csx.AddMetal("metal")
    primitive = metal.AddLinPoly(
        priority=1, points=points, norm_dir=norm_dir, elevation=elevation, length=length
    )
    return csx, primitive


class Unreadable:
    """A primitive whose vertices cannot be read."""

    def GetCoords(self):  # noqa: N802 - CSXCAD API
        raise RuntimeError("no coordinates")

    def GetTransform(self):  # noqa: N802 - CSXCAD API
        raise RuntimeError("no transform")


class CSPrimLinPoly(Unreadable):
    """Named like CSXCAD's class, so the helpers treat it as a polygon."""


class CSPrimPolyhedron:
    """A unit tetrahedron, named like CSXCAD's class."""

    VERTICES = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)]

    def GetTransform(self):  # noqa: N802 - CSXCAD API
        return None

    def GetNumVertices(self):  # noqa: N802 - CSXCAD API
        return len(self.VERTICES)

    def GetVertex(self, index):  # noqa: N802 - CSXCAD API
        return self.VERTICES[index]


class TestVertexBounds:
    @pytest.mark.parametrize(
        ("norm_dir", "flat_axis"), [(0, 0), (1, 1)], ids=["x-normal", "y-normal"]
    )
    def test_a_side_facing_polygon_sits_at_its_elevation(self, norm_dir, flat_axis):
        _csx, polygon = _metal_linpoly(
            [[0, 3, 3, 0], [0, 0, 2, 2]], norm_dir=norm_dir, elevation=5.0
        )

        bounds = fdtd_mesh._get_linpoly_vertex_bounds(polygon)

        assert set(bounds[flat_axis]) == {5.0}
        assert sorted(set(bounds[2])) == [0.0, 2.0]

    def test_unreadable_polygon_vertices_give_no_bounds(self):
        assert fdtd_mesh._get_linpoly_vertex_bounds(Unreadable()) == [[], [], []]

    def test_polyhedron_vertices_are_collected_per_axis(self):
        bounds = fdtd_mesh._get_polyhedron_vertex_bounds(CSPrimPolyhedron())

        assert bounds == [[0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]

    def test_a_transformed_polyhedron_is_moved(self):
        from CSXCAD.CSTransform import CSTransform

        class Moved(CSPrimPolyhedron):
            def GetTransform(self):  # noqa: N802 - CSXCAD API
                transform = CSTransform()
                transform.Translate([10, 0, 0])
                return transform

        bounds = fdtd_mesh._get_polyhedron_vertex_bounds(Moved())

        assert bounds[0] == pytest.approx([10, 14, 10, 10])

    def test_unreadable_polyhedron_vertices_give_no_bounds(self):
        assert fdtd_mesh._get_polyhedron_vertex_bounds(Unreadable()) == [[], [], []]


class TestCentroid:
    def test_polygon_centroid_is_the_vertex_mean(self):
        _csx, polygon = _metal_linpoly([[0, 4, 4, 0], [0, 0, 2, 2]], elevation=1.0)

        assert fdtd_mesh._prim_centroid(polygon) == pytest.approx([2.0, 1.0, 1.0])

    def test_polyhedron_centroid_is_the_vertex_mean(self):
        assert fdtd_mesh._prim_centroid(CSPrimPolyhedron()) == pytest.approx(
            [1.0, 1.0, 1.0]
        )

    def test_a_box_has_no_vertex_centroid(self):
        from CSXCAD import ContinuousStructure

        box = ContinuousStructure().AddMetal("m").AddBox([0, 0, 0], [1, 1, 1])

        assert fdtd_mesh._prim_centroid(box) is None

    def test_an_unreadable_polygon_has_no_centroid(self):
        assert fdtd_mesh._prim_centroid(CSPrimLinPoly()) is None


class TestCrossSection:
    def test_an_unreadable_polygon_has_no_cross_section(self):
        assert fdtd_mesh._linpoly_cross_section_point(CSPrimLinPoly(), 0, 1.0) is None

    def test_a_cut_outside_the_polygon_has_no_cross_section(self):
        _csx, polygon = _metal_linpoly([[0, 4, 4, 0], [0, 0, 2, 2]])

        assert fdtd_mesh._linpoly_cross_section_point(polygon, 0, 9.0) is None

    def test_interior_point_falls_back_to_a_cut_along_y(self, monkeypatch):
        _csx, polygon = _metal_linpoly([[0, 4, 4, 0], [0, 0, 2, 2]])
        original = fdtd_mesh._linpoly_cross_section_point
        monkeypatch.setattr(
            fdtd_mesh,
            "_linpoly_cross_section_point",
            lambda prim, dim, pos: None if dim == 0 else original(prim, dim, pos),
        )

        point = fdtd_mesh._linpoly_interior_xy(
            polygon, fdtd_mesh._get_prim_bounds(polygon)
        )

        assert point == pytest.approx([2.0, 1.0])

    def test_no_cut_through_the_polygon_gives_no_interior_point(self, monkeypatch):
        _csx, polygon = _metal_linpoly([[0, 4, 4, 0], [0, 0, 2, 2]])
        monkeypatch.setattr(
            fdtd_mesh, "_linpoly_cross_section_point", lambda prim, dim, pos: None
        )

        bounds = fdtd_mesh._get_prim_bounds(polygon)

        assert fdtd_mesh._linpoly_interior_xy(polygon, bounds) is None


class TestDecimateCurveCoords:
    def test_no_coordinates_give_none(self):
        assert fdtd_mesh._decimate_curve_coords([], 0.5) == []

    def test_the_last_vertex_replaces_one_too_close_to_it(self):
        """Both extremes are kept, so a kept vertex crowding the last one
        gives way to it."""
        kept = fdtd_mesh._decimate_curve_coords([0.0, 0.3, 0.6, 1.0], 0.5)

        assert kept == [0.0, 1.0]

    def test_a_close_pair_keeps_both_extremes(self):
        assert fdtd_mesh._decimate_curve_coords([0.2, 0.0], 0.5) == [0.0, 0.2]


class TestCollectAllBounds:
    @pytest.fixture
    def triangle(self):
        """Apex at x = 2: a vertex that is no edge of the bounding box."""
        return _metal_linpoly([[0, 4, 2], [0, 0, 3]])

    def test_an_apex_well_inside_gets_a_line(self, triangle):
        _csx, polygon = triangle

        bounds = fdtd_mesh._collect_all_bounds([polygon], [[], [], []], 0.5)

        assert bounds[0] == pytest.approx([0.0, 2.0, 4.0])

    def test_an_apex_crowding_an_edge_gets_none(self, triangle):
        _csx, polygon = triangle

        bounds = fdtd_mesh._collect_all_bounds([polygon], [[], [], []], 2.5)

        assert bounds[0] == pytest.approx([0.0, 4.0])

    def test_polyhedron_vertices_are_collected(self):
        from CSXCAD.CSTransform import CSTransform

        tetrahedron = type(
            "CSPrimPolyhedron",
            (CSPrimPolyhedron,),
            {
                "VERTICES": [(0, 0, 0), (4, 0, 0), (2, 3, 0), (2, 1, 2)],
                "GetBoundBox": lambda self: [[0, 0, 0], [4, 3, 2]],
                "GetTransform": lambda self: CSTransform(),
            },
        )()

        bounds = fdtd_mesh._collect_all_bounds([tetrahedron], [[], [], []], 0.5)

        assert bounds[0] == pytest.approx([0.0, 2.0, 4.0])
        assert bounds[1] == pytest.approx([0.0, 1.0, 3.0])


class TestTypeAtPosition:
    """A notched metal polygon only counts as metal where it has copper."""

    U_SHAPE = [[0, 3, 3, 2, 2, 1, 1, 0], [0, 0, 3, 3, 1, 1, 3, 3]]

    def test_metal_filling_the_notch_wins(self):
        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        metal = csx.AddMetal("metal")
        u_shape = metal.AddLinPoly(
            priority=1, points=self.U_SHAPE, norm_dir=2, elevation=0.0, length=0.035
        )
        plug = metal.AddLinPoly(
            priority=1,
            points=[[1, 2, 2, 1], [1, 1, 3, 3]],
            norm_dir=2,
            elevation=0.0,
            length=0.035,
        )

        for primitive in (u_shape, plug):
            primitive.Update()  # IsInside answers only once the shape is built

        assert fdtd_mesh._type_at_pos([u_shape, plug], 1, 2.0) == fdtd_mesh.Type.metal

    def test_a_primitive_that_cannot_be_probed_counts_as_covering(self):
        class CSPrimPolyhedron:
            def GetBoundBox(self):  # noqa: N802 - CSXCAD API
                return [[0, 0, 0], [1, 1, 1]]

            def GetTransform(self):  # noqa: N802 - CSXCAD API
                return None

            def GetProperty(self):  # noqa: N802 - CSXCAD API
                class Metal:
                    def GetTypeString(self):  # noqa: N802 - CSXCAD API
                        return "Metal"

                return Metal()

            def GetNumVertices(self):  # noqa: N802 - CSXCAD API
                return 0

            def IsInside(self, point):  # noqa: N802 - CSXCAD API
                raise RuntimeError("cannot probe")

        assert (
            fdtd_mesh._type_at_pos([CSPrimPolyhedron()], 0, 0.5) == fdtd_mesh.Type.metal
        )


# ---------------------------------------------------------------------
# Mesh internals, on a real meshed microstrip line
# ---------------------------------------------------------------------
@pytest.fixture
def line_mesh(built_mline):
    _line, sim, params, _ports = built_mline
    return fdtd_mesh.Mesh(sim.CSX, params), params


class TestSubstrateSpans:
    def test_a_dielectric_of_another_thickness_is_no_substrate(self, fr4, sim_for):
        """A via's antipad is a dielectric too, but only a copper layer thick."""
        from simpleEMS.components import GenericParams, GenericStructure

        params = GenericParams(
            min_freq=2e9,
            max_freq=3e9,
            target_freq=2.45e9,
            substrate_eps_r=4.4,
            substrate_tand=0.001,
            substrate_thickness_mm=1.6,
            substrate_width_mm=20.0,
            substrate_length_mm=30.0,
        )
        structure = GenericStructure(params, sim_for(params))
        structure.create_substrate()
        structure.create_ground()
        structure.create_via(
            position=(0.0, 0.0),
            via_diameter_mm=0.6,
            z_bottom_mm=-0.035,
            z_top_mm=1.6,
            antipad_diameter_mm=1.2,
            antipad_layers=[0.8],
        )

        mesh = fdtd_mesh.Mesh(structure.CSX, params)

        assert mesh.substrate_spans == [(0.0, 1.6)]

    def test_an_interval_outside_every_substrate_has_no_span(self, line_mesh):
        mesh, _params = line_mesh

        outside = BoundedType(Type.nonmetal, 50.0, 60.0)

        assert mesh._substrate_span_containing(2, outside) is None


class TestBoundedTypeLookups:
    def test_an_empty_dimension_becomes_air_across_the_box(self, line_mesh):
        mesh, _params = line_mesh

        expanded = mesh._set_expanded_bounds([[], [], []])

        for dim in range(3):
            (air,) = expanded[dim]
            assert air.get_type() == Type.air
            assert air.get_bounds() == pytest.approx(mesh._sim_box[dim])

    def test_nothing_ends_or_starts_at_an_unknown_position(self, line_mesh):
        mesh, _params = line_mesh

        assert mesh._type_below(0, 12345.0) is None
        assert mesh._type_above(0, 12345.0) is None
        assert mesh._type_below_meshed(0, 12345.0) is False
        assert mesh._type_above_meshed(0, 12345.0) is False


class TestLineGeneration:
    def test_a_sliver_collapses_to_its_midpoint(self, line_mesh):
        mesh, _params = line_mesh

        lines = mesh._regen_lines_or_collapse(
            1.0, 1.0001, 0.1, 0.1, 1.0, regen_threshold=0.01, dim=0, is_metal=True
        )

        assert lines == pytest.approx([1.00005])

    def test_a_wide_interval_is_regenerated(self, line_mesh):
        mesh, _params = line_mesh

        lines = mesh._regen_lines_or_collapse(
            0.0, 10.0, 0.5, 0.5, 1.0, regen_threshold=0.01, dim=0, is_metal=False
        )

        assert lines[0] == pytest.approx(0.0)
        assert lines[-1] == pytest.approx(10.0)

    def test_the_minimum_line_count_is_met_when_the_series_grow_fast(
        self, line_mesh, monkeypatch
    ):
        """Two geometric series grown from each end can span the interval in
        fewer steps than the minimum; both get padded."""
        mesh, _params = line_mesh
        unpadded = mesh._gen_lines_in_bounds(0.0, 20.0, 1.0, 1.0, 100.0, 0, False)
        monkeypatch.setattr(mesh, "_scaled_min_lines", lambda dist, metal, dim: 20)

        lines = mesh._gen_lines_in_bounds(0.0, 20.0, 1.0, 1.0, 100.0, 0, False)

        # Ten lines from each end, sharing the line where the halves meet.
        assert len(unpadded) < len(lines) == 19
        assert lines[0] == pytest.approx(0.0)
        assert lines[-1] == pytest.approx(20.0)


class TestCleanCloseLines:
    @staticmethod
    def x_lines_after_cleaning(mesh, lines, fixed):
        mesh.fixed_lines[0] = fixed
        grid = mesh.mesh
        grid.ClearLines(0)
        grid.AddLine("x", lines)

        mesh._clean_close_lines(min_spacing=0.01)

        return list(grid.GetLines(0))

    def test_two_free_lines_merge_at_their_midpoint(self, line_mesh):
        mesh, _params = line_mesh

        lines = self.x_lines_after_cleaning(mesh, [0.0, 0.002, 5.0], [])

        assert lines == pytest.approx([0.001, 5.0])

    def test_a_fixed_line_survives_a_later_neighbour(self, line_mesh):
        mesh, _params = line_mesh

        lines = self.x_lines_after_cleaning(mesh, [0.0, 0.002, 5.0], [0.0])

        assert lines == pytest.approx([0.0, 5.0])

    def test_a_fixed_line_replaces_an_earlier_neighbour(self, line_mesh):
        mesh, _params = line_mesh

        lines = self.x_lines_after_cleaning(mesh, [0.0, 0.002, 5.0], [0.002])

        assert lines == pytest.approx([0.002, 5.0])

    def test_a_dimension_with_one_line_is_left_alone(self, line_mesh):
        mesh, _params = line_mesh
        grid = mesh.mesh
        grid.ClearLines(1)
        grid.AddLine("y", [3.0])

        mesh._clean_close_lines(min_spacing=0.01)

        assert list(grid.GetLines(1)) == pytest.approx([3.0])
