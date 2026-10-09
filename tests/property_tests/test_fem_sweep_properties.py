"""Property tests for :mod:`simpleEMS.fem_sweep`.

The example tests drive the sweep with one fixed resonator. Here Hypothesis
draws the resonance -- where it sits in the band and how sharp it is -- and
checks the promises the sweep makes for any passive structure: it spends no
more than its solve ceiling, stays inside the band, never solves the same
frequency twice, hands back one S-matrix per output frequency, and keeps
the curve within ``_MAX_RESIDUAL`` of every solve point.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from simpleEMS.fem_sweep import _MAX_RESIDUAL, _passivity_excess, rational_sweep

BAND = np.linspace(2.0e9, 3.0e9, 201)

resonant_frequencies = st.floats(min_value=2.1e9, max_value=2.9e9)
quality_factors = st.floats(min_value=5.0, max_value=200.0)
budgets = st.integers(min_value=4, max_value=14)
finite_parts = st.floats(min_value=-10.0, max_value=10.0)


def _resonator(resonant_frequency: float, quality_factor: float):
    """A passive 1-port series RLC, like the shared ``synthetic_resonator``."""

    def s_at(frequency: float) -> np.ndarray:
        detune = quality_factor * (
            frequency / resonant_frequency - resonant_frequency / frequency
        )
        return np.array([[1.0 / (1.0 + 1j * detune) - 1.0]], dtype=complex)

    return s_at


class _CountingSolver:
    def __init__(self, s_at):
        self.s_at = s_at
        self.frequencies = []

    def __call__(self, frequency):
        self.frequencies.append(frequency)
        return self.s_at(frequency)


class TestPassivityExcess:
    @given(
        real=arrays(float, (8, 2, 2), elements=finite_parts),
        imag=arrays(float, (8, 2, 2), elements=finite_parts),
        scale=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_a_contraction_scores_zero(self, real, imag, scale):
        matrices = real + 1j * imag
        largest = np.linalg.svd(matrices, compute_uv=False).max()
        if largest > 0:
            matrices = matrices / largest * scale

        assert _passivity_excess(matrices) <= 1e-12

    @given(
        real=arrays(float, (8, 2, 2), elements=finite_parts),
        imag=arrays(float, (8, 2, 2), elements=finite_parts),
    )
    def test_the_excess_is_the_largest_singular_value_over_one(self, real, imag):
        matrices = real + 1j * imag
        largest = np.linalg.svd(matrices, compute_uv=False).max()

        assert _passivity_excess(matrices) == max(largest - 1.0, 0.0)


# A sweep runs dozens of AAA fits, so one example can take a few hundred ms
# on a busy machine; the deadline would only measure the machine
no_deadline = settings(deadline=None)


class TestRationalSweep:
    @no_deadline
    @given(
        resonant_frequency=resonant_frequencies,
        quality_factor=quality_factors,
        budget=budgets,
    )
    def test_solves_stay_within_the_ceiling_band_and_are_unique(
        self, resonant_frequency, quality_factor, budget
    ):
        solver = _CountingSolver(_resonator(resonant_frequency, quality_factor))

        model = rational_sweep(BAND, [1], solver, budget, verbose=False)

        assert model.shape == (len(BAND), 1, 1)
        # the default ceiling is twice the budget
        assert len(solver.frequencies) <= 2 * budget
        assert min(solver.frequencies) >= BAND[0]
        assert max(solver.frequencies) <= BAND[-1]
        assert len(set(solver.frequencies)) == len(solver.frequencies)

    @no_deadline
    @given(
        resonant_frequency=resonant_frequencies,
        quality_factor=quality_factors,
        budget=budgets,
    )
    def test_the_curve_stays_close_to_its_solve_points(
        self, resonant_frequency, quality_factor, budget
    ):
        s_at = _resonator(resonant_frequency, quality_factor)
        solver = _CountingSolver(s_at)
        model = rational_sweep(BAND, [1], solver, budget, verbose=False)

        # The sweep may trade exact interpolation for passivity, but never by
        # more than _MAX_RESIDUAL; check it wherever a solve hit the grid
        for frequency in solver.frequencies:
            index = int(np.argmin(np.abs(BAND - frequency)))
            if np.isclose(BAND[index], frequency, rtol=0, atol=1.0):
                np.testing.assert_allclose(
                    model[index], s_at(frequency), atol=_MAX_RESIDUAL
                )
