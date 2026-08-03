"""Unit tests for :mod:`simpleEMS.filter_coefficient`.

The reference g-values are Pozar, *Microwave Engineering* (4th ed.), Tables
8.3 (Butterworth / maximally flat) and 8.4 (equal-ripple Chebyshev). Bessel
values come from the module's own hard-coded table, so those tests pin its
structure and boundaries rather than re-deriving it.
"""

import math

import pytest

from simpleEMS.filter_coefficient import (
    BESSEL_COEFFICIENT,
    bessel_value,
    butterworth_value,
    chebyshev_value,
    get_filter_coefficient,
)


# ---------------------------------------------------------------------
# Butterworth
# ---------------------------------------------------------------------
class TestButterworth:
    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (1, [2.0000]),
            (2, [1.4142, 1.4142]),
            (3, [1.0000, 2.0000, 1.0000]),
            (4, [0.7654, 1.8478, 1.8478, 0.7654]),
            (5, [0.6180, 1.6180, 2.0000, 1.6180, 0.6180]),
            (6, [0.5176, 1.4142, 1.9319, 1.9319, 1.4142, 0.5176]),
        ],
    )
    def test_matches_pozar_table_8_3(self, order, expected):
        values = [butterworth_value(i, order) for i in range(order)]

        assert values == pytest.approx(expected, abs=1e-4)

    def test_matches_closed_form(self):
        order = 7
        for i in range(order):
            expected = 2.0 * math.sin((2.0 * i + 1.0) / (2.0 * order) * math.pi)
            assert butterworth_value(i, order) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("order", [2, 3, 4, 5, 6, 7, 8])
    def test_coefficients_are_symmetric(self, order):
        """A doubly-terminated maximally-flat prototype is a palindrome."""
        values = [butterworth_value(i, order) for i in range(order)]

        assert values == pytest.approx(values[::-1], rel=1e-12)

    @pytest.mark.parametrize("index", [-1, -5, 3, 99])
    def test_out_of_range_index_returns_unity(self, index):
        """Index 0 and index n both denote a termination, i.e. g = 1."""
        assert butterworth_value(index, 3) == 1.0

    def test_all_values_positive(self):
        assert all(butterworth_value(i, 9) > 0 for i in range(9))


# ---------------------------------------------------------------------
# Chebyshev
# ---------------------------------------------------------------------
class TestChebyshev:
    @pytest.mark.parametrize(
        ("order", "ripple", "expected"),
        [
            (3, 0.1, [1.0316, 1.1474, 1.0316]),
            (5, 0.1, [1.1468, 1.3712, 1.9750, 1.3712, 1.1468]),
            (7, 0.1, [1.1812, 1.4228, 2.0967, 1.5734, 2.0967, 1.4228, 1.1812]),
            (3, 0.5, [1.5963, 1.0967, 1.5963]),
            (5, 0.5, [1.7058, 1.2296, 2.5408, 1.2296, 1.7058]),
        ],
    )
    def test_matches_pozar_table_8_4(self, order, ripple, expected):
        values = [chebyshev_value(i, order, ripple) for i in range(order)]

        assert values == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("order", [3, 5, 7, 9])
    def test_coefficients_are_symmetric(self, order):
        values = [chebyshev_value(i, order, 0.1) for i in range(order)]

        assert values == pytest.approx(values[::-1], rel=1e-9)

    def test_larger_ripple_gives_larger_first_element(self):
        """More passband ripple buys a steeper skirt at the cost of match."""
        values = [chebyshev_value(0, 3, r) for r in (0.01, 0.1, 0.5, 1.0)]

        assert values == sorted(values)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_even_order_raises(self, order):
        """Equal-ripple even orders need unequal terminations, so they cannot
        be realized passively here."""
        with pytest.raises(ValueError, match="Even order"):
            chebyshev_value(0, order, 0.1)

    @pytest.mark.parametrize("index", [-1, 3, 50])
    def test_out_of_range_index_returns_unity_before_order_check(self, index):
        """The index guard runs first, so an even order never raises for an
        out-of-range index."""
        assert chebyshev_value(index, 3, 0.1) == 1.0

    def test_all_values_positive(self):
        assert all(chebyshev_value(i, 7, 0.1) > 0 for i in range(7))


# ---------------------------------------------------------------------
# Bessel
# ---------------------------------------------------------------------
class TestBessel:
    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (2, [0.5755, 2.1478]),
            (3, [0.3374, 0.9705, 2.2034]),
            (5, [0.1743, 0.5072, 0.8040, 1.1110, 2.2582]),
        ],
    )
    def test_matches_stored_table(self, order, expected):
        values = [bessel_value(i, order) for i in range(order)]

        assert values == pytest.approx(expected, abs=1e-4)

    @pytest.mark.parametrize("order", range(2, 20))
    def test_every_supported_order_has_a_full_row(self, order):
        """The table is indexed ``order - 2``; a short or missing row would
        silently return a neighbouring order's coefficients."""
        row = BESSEL_COEFFICIENT[order - 2]

        assert len(row) == order
        assert all(value > 0 for value in row)

    @pytest.mark.parametrize("order", [0, 1, 20, 50, -3])
    def test_unsupported_order_raises(self, order):
        with pytest.raises(ValueError, match="not supported"):
            bessel_value(0, order)

    @pytest.mark.parametrize("index", [-1, 3, 99])
    def test_out_of_range_index_returns_unity(self, index):
        assert bessel_value(index, 3) == 1.0

    def test_table_covers_exactly_orders_2_to_19(self):
        assert len(BESSEL_COEFFICIENT) == 18


# ---------------------------------------------------------------------
# get_filter_coefficient dispatch
# ---------------------------------------------------------------------
class TestGetFilterCoefficient:
    def test_dispatches_to_bessel(self):
        assert get_filter_coefficient(0, "bessel", 3, None) == bessel_value(0, 3)

    def test_dispatches_to_butterworth(self):
        assert get_filter_coefficient(0, "butterworth", 3, None) == butterworth_value(
            0, 3
        )

    def test_dispatches_to_chebyshev(self):
        assert get_filter_coefficient(0, "chebyshev", 3, 0.5) == chebyshev_value(
            0, 3, 0.5
        )

    @pytest.mark.parametrize(
        "response", ["BESSEL", "Butterworth", "ChebyShev", "CHEBYSHEV"]
    )
    def test_response_name_is_case_insensitive(self, response):
        value = get_filter_coefficient(0, response, 3, 0.1)

        assert value > 0

    def test_none_ripple_defaults_to_0_1_db(self):
        """Documented default; a different default would silently change every
        Chebyshev filter built without an explicit ripple."""
        defaulted = get_filter_coefficient(0, "chebyshev", 3, None)

        assert defaulted == pytest.approx(chebyshev_value(0, 3, 0.1), rel=1e-12)

    def test_ripple_is_ignored_for_non_chebyshev(self):
        a = get_filter_coefficient(0, "butterworth", 3, 0.1)
        b = get_filter_coefficient(0, "butterworth", 3, 3.0)

        assert a == b

    @pytest.mark.parametrize("response", ["elliptic", "gaussian", "", "cheby"])
    def test_unknown_response_raises(self, response):
        with pytest.raises(ValueError, match="Unsupported filter response"):
            get_filter_coefficient(0, response, 3, None)

    def test_underlying_order_errors_propagate(self):
        with pytest.raises(ValueError, match="Even order"):
            get_filter_coefficient(0, "chebyshev", 4, 0.1)

        with pytest.raises(ValueError, match="not supported"):
            get_filter_coefficient(0, "bessel", 25, None)
