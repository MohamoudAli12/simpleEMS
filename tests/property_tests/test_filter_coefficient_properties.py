"""Property tests for :mod:`simpleEMS.filter_coefficient`.

A low-pass prototype with equal source and load terminations has a
symmetric ladder, so its g-values read the same from either end. That holds
for every Butterworth order and every odd-order Chebyshev, whatever the
ripple, which makes it a good check on the closed-form recursions.
"""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from simpleEMS.filter_coefficient import (
    bessel_value,
    butterworth_value,
    chebyshev_value,
)

orders = st.integers(min_value=1, max_value=19)
odd_orders = st.integers(min_value=0, max_value=9).map(lambda half: 2 * half + 1)
bessel_orders = st.integers(min_value=2, max_value=19)
ripples_db = st.floats(min_value=0.01, max_value=3.0)
any_index = st.integers(min_value=-5, max_value=40)


class TestButterworth:
    @given(order=orders, data=st.data())
    def test_g_values_are_symmetric(self, order, data):
        index = data.draw(st.integers(0, order - 1))

        assert butterworth_value(index, order) == pytest.approx(
            butterworth_value(order - 1 - index, order)
        )

    @given(order=orders, data=st.data())
    def test_g_values_lie_in_zero_to_two(self, order, data):
        index = data.draw(st.integers(0, order - 1))

        assert 0.0 < butterworth_value(index, order) <= 2.0

    @given(order=orders, index=any_index)
    def test_out_of_range_index_terminates_at_one(self, order, index):
        assume(index < 0 or index >= order)

        assert butterworth_value(index, order) == 1.0


class TestChebyshev:
    @given(order=odd_orders, ripple_db=ripples_db, data=st.data())
    def test_odd_order_g_values_are_symmetric(self, order, ripple_db, data):
        index = data.draw(st.integers(0, order - 1))

        assert chebyshev_value(index, order, ripple_db) == pytest.approx(
            chebyshev_value(order - 1 - index, order, ripple_db), rel=1e-9
        )

    @given(order=odd_orders, ripple_db=ripples_db, data=st.data())
    def test_g_values_are_positive_and_finite(self, order, ripple_db, data):
        index = data.draw(st.integers(0, order - 1))
        value = chebyshev_value(index, order, ripple_db)

        assert value > 0.0
        assert math.isfinite(value)

    @given(
        order=st.integers(min_value=1, max_value=10).map(lambda half: 2 * half),
        ripple_db=ripples_db,
        data=st.data(),
    )
    def test_even_orders_are_rejected(self, order, ripple_db, data):
        index = data.draw(st.integers(0, order - 1))

        with pytest.raises(ValueError, match="Even order"):
            chebyshev_value(index, order, ripple_db)

    @given(order=odd_orders, ripple_db=ripples_db, index=any_index)
    def test_out_of_range_index_terminates_at_one(self, order, ripple_db, index):
        assume(index < 0 or index >= order)

        assert chebyshev_value(index, order, ripple_db) == 1.0


class TestBessel:
    @given(order=bessel_orders, data=st.data())
    def test_tabulated_g_values_are_positive(self, order, data):
        index = data.draw(st.integers(0, order - 1))

        assert bessel_value(index, order) > 0.0

    @given(order=bessel_orders, index=any_index)
    def test_out_of_range_index_terminates_at_one(self, order, index):
        assume(index < 0 or index >= order)

        assert bessel_value(index, order) == 1.0

    @given(
        order=st.integers(min_value=-5, max_value=40).filter(
            lambda order: not 2 <= order <= 19
        )
    )
    def test_unsupported_orders_are_rejected(self, order):
        with pytest.raises(ValueError, match="not supported"):
            bessel_value(0, order)
