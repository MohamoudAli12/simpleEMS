"""Tests for :mod:`simpleEMS.fem_sweep` -- the adaptive rational sweep.

``rational_sweep`` is what makes the FEM backend affordable: it solves at a
handful of frequencies and interpolates the rest. Driving it with an analytic
S-parameter function instead of a real solve makes the whole algorithm --
point placement, order selection, passivity scoring -- testable in
milliseconds and with an exact answer to compare against.
"""

import numpy as np
import pytest

from simpleEMS.fem_sweep import _fit_matrix, _passivity_excess, rational_sweep


FGRID = np.linspace(2.0e9, 3.0e9, 401)


def _three_resonances(freq: float) -> np.ndarray:
    """A passive 1-port response with three poles in band.

    Higher order than the single-resonance fixture, so a small solve budget
    genuinely cannot fit it -- which is what makes convergence measurable.
    """
    total = 0.0
    for f0, q in ((2.2e9, 60.0), (2.45e9, 90.0), (2.8e9, 50.0)):
        detune = q * (freq / f0 - f0 / freq)
        total += 0.33 * (1.0 / (1.0 + 1j * detune) - 1.0)
    return np.array([[total]], dtype=complex)


@pytest.fixture
def counting_solver(synthetic_resonator):
    """Wrap the analytic resonator so tests can count and inspect solves."""

    class Solver:
        def __init__(self):
            self.frequencies = []

        def __call__(self, freq):
            self.frequencies.append(freq)
            return synthetic_resonator(freq)

    return Solver()


# ---------------------------------------------------------------------
# _passivity_excess
# ---------------------------------------------------------------------
class TestPassivityExcess:
    def test_passive_model_scores_zero(self):
        model = np.full((10, 1, 1), 0.5 + 0j)

        assert _passivity_excess(model) == 0.0

    def test_unit_magnitude_is_exactly_passive(self):
        """A lossless total reflection sits on the passivity boundary."""
        model = np.full((10, 1, 1), 1.0 + 0j)

        assert _passivity_excess(model) == pytest.approx(0.0, abs=1e-12)

    def test_active_model_reports_the_excess(self):
        model = np.full((10, 1, 1), 1.3 + 0j)

        assert _passivity_excess(model) == pytest.approx(0.3, rel=1e-9)

    def test_reports_the_worst_point_over_frequency(self):
        model = np.ones((5, 1, 1), dtype=complex) * 0.2
        model[3, 0, 0] = 1.75

        assert _passivity_excess(model) == pytest.approx(0.75, rel=1e-9)

    def test_uses_the_largest_singular_value_for_multiport(self):
        """For a matrix it is the largest singular value, not the largest
        entry, that must stay below 1."""
        model = np.zeros((1, 2, 2), dtype=complex)
        model[0] = np.array([[0.8, 0.8], [0.0, 0.0]])
        expected = np.linalg.svd(model[0], compute_uv=False).max() - 1.0

        assert _passivity_excess(model) == pytest.approx(max(expected, 0.0), rel=1e-9)


# ---------------------------------------------------------------------
# _fit_matrix
# ---------------------------------------------------------------------
class TestFitMatrix:
    def test_reproduces_a_rational_function_exactly(self):
        """AAA is exact for rational data once it has enough terms."""
        zs = np.linspace(-1, 1, 12)
        s_arr = (1.0 / (zs + 2.0)).reshape(-1, 1, 1).astype(complex)
        zg = np.linspace(-1, 1, 60)

        model = _fit_matrix(zs, s_arr, zg, max_terms=12)
        truth = (1.0 / (zg + 2.0)).reshape(-1, 1, 1)

        assert model == pytest.approx(truth, abs=1e-10)

    def test_output_shape_follows_the_evaluation_grid(self):
        zs = np.linspace(-1, 1, 8)
        s_arr = np.random.default_rng(0).normal(size=(8, 2, 2)).astype(complex)
        zg = np.linspace(-1, 1, 33)

        model = _fit_matrix(zs, s_arr, zg, max_terms=8)

        assert model.shape == (33, 2, 2)
        assert model.dtype == complex

    def test_fits_each_matrix_entry_independently(self):
        """S11 and S21 have different shapes; mixing them up would show as one
        entry leaking into the other."""
        zs = np.linspace(-1, 1, 10)
        s_arr = np.zeros((10, 2, 2), dtype=complex)
        s_arr[:, 0, 0] = 1.0 / (zs + 3.0)
        s_arr[:, 1, 0] = zs**2

        model = _fit_matrix(zs, s_arr, zs, max_terms=10)

        assert model[:, 0, 0] == pytest.approx(s_arr[:, 0, 0], abs=1e-9)
        assert model[:, 1, 0] == pytest.approx(s_arr[:, 1, 0], abs=1e-9)
        assert model[:, 0, 1] == pytest.approx(np.zeros(10), abs=1e-9)

    def test_low_order_cap_does_not_raise(self):
        """Capping ``max_terms`` below what AAA needs is deliberate; the
        convergence warning must stay suppressed rather than surfacing."""
        zs = np.linspace(-1, 1, 12)
        s_arr = (1.0 / (zs + 1.01)).reshape(-1, 1, 1).astype(complex)

        model = _fit_matrix(zs, s_arr, zs, max_terms=2)

        assert np.all(np.isfinite(model))


# ---------------------------------------------------------------------
# rational_sweep
# ---------------------------------------------------------------------
class TestRationalSweep:
    def test_recovers_an_analytic_resonance(self, counting_solver, synthetic_resonator):
        """The end-to-end claim: ~14 solves reproduce a 401-point curve."""
        model = rational_sweep(FGRID, [1], counting_solver, 14, verbose=False)
        truth = np.array([synthetic_resonator(f) for f in FGRID])

        assert model == pytest.approx(truth, abs=1e-6)

    def test_finds_the_resonant_frequency(self, counting_solver):
        model = rational_sweep(FGRID, [1], counting_solver, 14, verbose=False)

        resonance = FGRID[np.argmin(np.abs(model[:, 0, 0]))]

        assert resonance == pytest.approx(2.45e9, rel=1e-3)

    def test_output_shape_matches_grid_and_port_count(self, counting_solver):
        model = rational_sweep(FGRID, [1], counting_solver, 8, verbose=False)

        assert model.shape == (len(FGRID), 1, 1)

    def test_two_port_shape(self):
        def solve(freq):
            return np.array([[0.1 + 0j, 0.9 + 0j], [0.9 + 0j, 0.1 + 0j]])

        model = rational_sweep(FGRID, [1, 2], solve, 6, verbose=False)

        assert model.shape == (len(FGRID), 2, 2)

    def test_respects_the_solve_budget(self, counting_solver):
        rational_sweep(FGRID, [1], counting_solver, 9, verbose=False)

        assert len(counting_solver.frequencies) == 9

    @pytest.mark.parametrize("budget", [4, 5, 7, 12])
    def test_never_exceeds_the_budget(self, counting_solver, budget):
        rational_sweep(FGRID, [1], counting_solver, budget, verbose=False)

        assert len(counting_solver.frequencies) <= budget

    def test_seeds_with_uniform_points_before_adapting(self, counting_solver):
        """The first five solves are the evenly-spaced seed; only after that
        does point placement become adaptive."""
        rational_sweep(FGRID, [1], counting_solver, 10, verbose=False)

        seed = counting_solver.frequencies[:5]

        assert seed == pytest.approx(np.linspace(2.0e9, 3.0e9, 5), rel=1e-9)

    def test_solves_stay_inside_the_requested_range(self, counting_solver):
        rational_sweep(FGRID, [1], counting_solver, 12, verbose=False)

        assert min(counting_solver.frequencies) >= FGRID[0]
        assert max(counting_solver.frequencies) <= FGRID[-1]

    def test_adaptive_points_cluster_near_the_resonance(self, counting_solver):
        """Point selection weights fast-changing regions, so the added solves
        should land nearer 2.45 GHz than a uniform grid would."""
        rational_sweep(FGRID, [1], counting_solver, 14, verbose=False)

        adaptive = np.array(counting_solver.frequencies[5:])
        uniform = np.linspace(2.0e9, 3.0e9, len(adaptive))

        assert np.mean(np.abs(adaptive - 2.45e9)) < np.mean(np.abs(uniform - 2.45e9))

    def test_never_solves_the_same_frequency_twice(self, counting_solver):
        rational_sweep(FGRID, [1], counting_solver, 16, verbose=False)

        assert len(set(counting_solver.frequencies)) == len(counting_solver.frequencies)

    def test_result_is_passive_for_a_passive_structure(self, counting_solver):
        model = rational_sweep(FGRID, [1], counting_solver, 14, verbose=False)

        assert _passivity_excess(model) < 1e-6

    def test_more_solves_give_a_better_fit(self):
        """Convergence check on a target too rich for a handful of solves.

        A single resonance is a low-order rational function that even 6 solves
        reproduce to machine precision, which would make this vacuous -- hence
        the three-pole response.
        """
        truth = np.array([_three_resonances(f) for f in FGRID])

        coarse = rational_sweep(FGRID, [1], _three_resonances, 6, verbose=False)
        fine = rational_sweep(FGRID, [1], _three_resonances, 16, verbose=False)

        coarse_err = np.max(np.abs(coarse - truth))
        fine_err = np.max(np.abs(fine - truth))

        assert fine_err < coarse_err / 100
        assert fine_err < 1e-6

    def test_tolerance_can_stop_the_sweep_early(self, counting_solver):
        """A flat response converges immediately, so a loose tolerance should
        cut the sweep short."""

        def flat(freq):
            return np.array([[0.25 + 0j]])

        calls = []

        def counted(freq):
            calls.append(freq)
            return flat(freq)

        rational_sweep(FGRID, [1], counted, 30, tol=1e-3, verbose=False)

        assert len(calls) < 30

    def test_zero_tolerance_uses_the_whole_budget(self, counting_solver):
        """``tol=0`` is documented as disabling the early-exit check."""
        rational_sweep(FGRID, [1], counting_solver, 11, tol=0.0, verbose=False)

        assert len(counting_solver.frequencies) == 11

    def test_single_point_grid_does_not_crash(self):
        """A one-point grid has no interval for ``pick_next`` to bisect.

        Regression guard: this used to raise ``ValueError: attempt to get
        argmax of an empty sequence`` from inside ``pick_next``, since
        ``fgrid[:-1]`` is empty for a single point. Reachable via
        ``num_points=1`` on the FEM backend.
        """
        grid = np.array([2.45e9])

        model = rational_sweep(
            grid, [1], lambda f: np.array([[0.3 + 0j]]), 4, verbose=False
        )

        assert model.shape == (1, 1, 1)
        assert np.all(np.isfinite(model))

    def test_single_point_grid_returns_the_solved_value(self):
        """The seeded solve is the whole answer, so it must survive intact."""
        grid = np.array([2.45e9])

        model = rational_sweep(
            grid, [1], lambda f: np.array([[0.3 + 0.4j]]), 4, verbose=False
        )

        assert model[0, 0, 0] == pytest.approx(0.3 + 0.4j, abs=1e-9)

    def test_single_point_grid_solves_exactly_once(self):
        """One output point costs one solve, whatever the budget says.

        The seeding loop used to spread ``min(5, num_solves)`` points across
        the band; with a degenerate band ``np.linspace(f, f, 5)`` repeated the
        same frequency five times and threw four full solves away.
        """
        calls = []

        def counted(freq):
            calls.append(freq)
            return np.array([[0.3 + 0j]])

        rational_sweep(np.array([2.45e9]), [1], counted, 10, verbose=False)

        assert calls == [2.45e9]

    def test_verbose_output_mentions_the_solves(self, counting_solver, capsys):
        rational_sweep(FGRID, [1], counting_solver, 6, verbose=True)

        out = capsys.readouterr().out

        assert "FEM solved" in out
        assert "GHz" in out
