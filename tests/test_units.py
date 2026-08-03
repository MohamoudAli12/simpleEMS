"""Unit tests for the length-conversion helpers in :mod:`simpleEMS.sim_tools`.

Trivial functions, but every geometry calculation routes through them and a
sign or factor slip would be invisible in a plot. Round-trip identities catch
inverted pairs; the literal values pin the factors themselves.
"""

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import (  # noqa: E402
    cm_to_mm,
    m_to_mm,
    mil_to_mm,
    mm_to_cm,
    mm_to_m,
    mm_to_mil,
)


CONVERSION_PAIRS = [
    (mm_to_m, m_to_mm),
    (m_to_mm, mm_to_m),
    (mil_to_mm, mm_to_mil),
    (mm_to_mil, mil_to_mm),
    (cm_to_mm, mm_to_cm),
    (mm_to_cm, cm_to_mm),
]


class TestKnownValues:
    @pytest.mark.parametrize(
        ("func", "value", "expected"),
        [
            (mm_to_m, 1000.0, 1.0),
            (mm_to_m, 1.6, 0.0016),
            (m_to_mm, 1.0, 1000.0),
            (m_to_mm, 0.0016, 1.6),
            (mil_to_mm, 1.0, 0.0254),
            (mil_to_mm, 1000.0, 25.4),
            (mm_to_mil, 25.4, 1000.0),
            (mm_to_mil, 0.0254, 1.0),
            (cm_to_mm, 1.0, 10.0),
            (cm_to_mm, 2.54, 25.4),
            (mm_to_cm, 10.0, 1.0),
            (mm_to_cm, 25.4, 2.54),
        ],
    )
    def test_exact_factors(self, func, value, expected):
        assert func(value) == pytest.approx(expected, rel=1e-12)

    def test_one_inch_chain(self):
        """1000 mil -> 25.4 mm -> 2.54 cm -> 0.0254 m."""
        mm = mil_to_mm(1000.0)

        assert mm == pytest.approx(25.4, rel=1e-12)
        assert mm_to_cm(mm) == pytest.approx(2.54, rel=1e-12)
        assert mm_to_m(mm) == pytest.approx(0.0254, rel=1e-12)


class TestRoundTrips:
    @pytest.mark.parametrize(("forward", "backward"), CONVERSION_PAIRS)
    @pytest.mark.parametrize("value", [0.0, 1.0, 1.6, 37.234, 1e-6, 1e6, -5.5])
    def test_round_trip_is_identity(self, forward, backward, value):
        assert backward(forward(value)) == pytest.approx(value, rel=1e-12, abs=1e-15)


class TestBehaviour:
    @pytest.mark.parametrize(
        "func", [mm_to_m, m_to_mm, mil_to_mm, mm_to_mil, cm_to_mm, mm_to_cm]
    )
    def test_zero_maps_to_zero(self, func):
        assert func(0.0) == 0.0

    @pytest.mark.parametrize(
        "func", [mm_to_m, m_to_mm, mil_to_mm, mm_to_mil, cm_to_mm, mm_to_cm]
    )
    def test_sign_is_preserved(self, func):
        assert func(-3.0) < 0

    @pytest.mark.parametrize(
        "func", [mm_to_m, m_to_mm, mil_to_mm, mm_to_mil, cm_to_mm, mm_to_cm]
    )
    def test_conversions_are_linear(self, func):
        """Scaling the input scales the output by the same factor."""
        assert func(6.0) == pytest.approx(3 * func(2.0), rel=1e-12)

    @pytest.mark.parametrize(
        "func", [mm_to_m, m_to_mm, mil_to_mm, mm_to_mil, cm_to_mm, mm_to_cm]
    )
    def test_accepts_numpy_arrays(self, func):
        """Params classes pass arrays (e.g. the thirds rule) through these."""
        values = np.array([1.0, 2.0, 3.0])

        result = func(values)

        assert isinstance(result, np.ndarray)
        assert result == pytest.approx([func(1.0), func(2.0), func(3.0)], rel=1e-12)
