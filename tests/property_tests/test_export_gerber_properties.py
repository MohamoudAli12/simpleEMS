"""Property tests for :mod:`simpleEMS.export_gerber`.

Two helpers decide whether a Gerber file is right before any geometry is
involved: the coordinate formatter, which must place every point within
half a unit of the declared 1e-5 mm resolution, and the interval merge
that groups copper by height into layers.
"""

import re
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

# simpleEMS imports CSXCAD at module scope, so without it this module cannot
# even be collected.
pytest.importorskip("CSXCAD")

from simpleEMS.export_gerber import (  # noqa: E402
    _EPS,
    _merge_intervals,
    gerber_coord,
)

GERBER_UNIT_MM = 1e-5
COORDINATE = re.compile(r"X(-?\d+)Y(-?\d+)")

# A board fits comfortably inside a metre
positions_mm = st.floats(min_value=-1000.0, max_value=1000.0)
heights_mm = st.floats(min_value=-10.0, max_value=10.0)
intervals = st.tuples(heights_mm, st.floats(min_value=0.0, max_value=5.0)).map(
    lambda start_and_length: (
        start_and_length[0],
        start_and_length[0] + start_and_length[1],
    )
)


class TestGerberCoord:
    @given(positions_mm, positions_mm)
    def test_the_coordinate_round_trips_to_within_half_a_unit(self, x_pos, y_pos):
        match = COORDINATE.fullmatch(gerber_coord((x_pos, y_pos)))

        assert match is not None
        assert int(match[1]) * GERBER_UNIT_MM == pytest.approx(
            x_pos, abs=GERBER_UNIT_MM / 2 + 1e-12
        )
        assert int(match[2]) * GERBER_UNIT_MM == pytest.approx(
            y_pos, abs=GERBER_UNIT_MM / 2 + 1e-12
        )

    @given(positions_mm, positions_mm)
    def test_the_format_has_no_padding_or_plus_sign(self, x_pos, y_pos):
        coordinate = gerber_coord((x_pos, y_pos))

        assert "+" not in coordinate
        assert not re.search(r"[XY]-?0\d", coordinate)


class TestMergeIntervals:
    @given(st.lists(intervals, max_size=20))
    def test_merged_intervals_are_sorted_and_separated(self, spans):
        merged = _merge_intervals(spans)

        for lower, upper in merged:
            assert lower <= upper
        for (_, previous_top), (next_bottom, _) in pairwise(merged):
            assert next_bottom > previous_top + _EPS

    @given(st.lists(intervals, max_size=20))
    def test_every_input_lies_inside_one_merged_interval(self, spans):
        merged = _merge_intervals(spans)

        for bottom, top in spans:
            assert any(lower <= bottom and top <= upper for lower, upper in merged)

    @given(st.lists(intervals, max_size=20))
    def test_merged_intervals_start_and_end_on_input_endpoints(self, spans):
        bottoms = {bottom for bottom, _ in spans}
        tops = {top for _, top in spans}

        for lower, upper in _merge_intervals(spans):
            assert lower in bottoms
            assert upper in tops

    @given(st.lists(intervals, max_size=20))
    def test_merging_twice_changes_nothing(self, spans):
        merged = _merge_intervals(spans)

        assert _merge_intervals([tuple(span) for span in merged]) == merged
