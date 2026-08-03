"""Tests for the objective functions, sweep, and optimizer in ``sim_tools``.

``optimize_s11``/``optimize_s21`` are pure functions of an S-parameter array,
so they are tested directly against hand-computed dB values. ``param_sweep``
and ``optimize_s_params`` take a user-supplied ``simulate_fn``, which makes
them testable with a cheap analytic stand-in instead of a solver -- the thing
under test is the orchestration (how many calls, with what arguments, in what
order), not the electromagnetics.
"""

from pathlib import Path

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import (  # noqa: E402
    SimData,
    optimize_s11,
    optimize_s21,
    optimize_s_params,
    param_sweep,
)


FREQS = np.linspace(1e9, 3e9, 201)


def s_from_db(db_values):
    """Build a complex S-parameter array with the given magnitudes in dB."""
    return 10 ** (np.asarray(db_values) / 20.0) + 0j


def notch_at(centre, depth_db=-30.0, width=0.2e9):
    """An S11 curve with a single well-matched notch."""
    envelope = depth_db * np.exp(-(((FREQS - centre) / width) ** 2))
    return s_from_db(envelope)


# ---------------------------------------------------------------------
# optimize_s11
# ---------------------------------------------------------------------
class TestOptimizeS11:
    def test_target_freq_returns_the_value_at_that_point(self):
        s11 = notch_at(2.45e9)
        index = np.abs(FREQS - 2.45e9).argmin()
        expected = 20 * np.log10(np.abs(s11[index]))

        assert optimize_s11(FREQS, s11, target_freq=2.45e9) == pytest.approx(expected)

    def test_target_freq_snaps_to_the_nearest_grid_point(self):
        """The requested frequency need not be on the grid."""
        s11 = notch_at(2.45e9)

        exact = optimize_s11(FREQS, s11, target_freq=2.45e9)
        offset = optimize_s11(FREQS, s11, target_freq=2.4500001e9)

        assert exact == offset

    def test_target_freq_takes_precedence_over_freq_band(self):
        """Documented: when both are given, ``freq_band``, ``mode`` and
        ``threshold`` are ignored."""
        s11 = notch_at(2.45e9)

        with_band = optimize_s11(
            FREQS, s11, target_freq=2.45e9, freq_band=(1e9, 3e9), mode="mean"
        )
        without = optimize_s11(FREQS, s11, target_freq=2.45e9)

        assert with_band == without

    def test_worst_mode_returns_the_maximum_in_band(self):
        s11 = s_from_db(np.linspace(-30, -3, len(FREQS)))

        assert optimize_s11(
            FREQS, s11, freq_band=(1e9, 3e9), mode="worst"
        ) == pytest.approx(-3, abs=0.1)

    def test_mean_mode_returns_the_band_average(self):
        s11 = s_from_db(np.full(len(FREQS), -20.0))

        assert optimize_s11(
            FREQS, s11, freq_band=(1e9, 3e9), mode="mean"
        ) == pytest.approx(-20.0, abs=1e-9)

    def test_threshold_mode_is_zero_when_fully_below_threshold(self):
        """Nothing exceeds -15 dB, so there is no penalty to pay."""
        s11 = s_from_db(np.full(len(FREQS), -30.0))

        assert optimize_s11(
            FREQS, s11, freq_band=(1e9, 3e9), mode="threshold", threshold=-15
        ) == pytest.approx(0.0)

    def test_threshold_mode_accumulates_the_excess(self):
        s11 = s_from_db(np.full(len(FREQS), -10.0))
        in_band = np.sum((FREQS >= 1e9) & (FREQS <= 3e9))

        penalty = optimize_s11(
            FREQS, s11, freq_band=(1e9, 3e9), mode="threshold", threshold=-15
        )

        assert penalty == pytest.approx(5.0 * in_band, rel=1e-6)

    def test_band_restricts_the_evaluation(self):
        """A bad match outside the band of interest must not be counted."""
        s11 = s_from_db(np.where(FREQS < 2e9, -3.0, -30.0))

        assert optimize_s11(
            FREQS, s11, freq_band=(2.5e9, 3e9), mode="worst"
        ) == pytest.approx(-30.0, abs=0.1)

    def test_lower_cost_means_a_better_match(self):
        """The optimizer minimises this, so the ordering has to be right."""
        good = optimize_s11(FREQS, notch_at(2.45e9, -30), target_freq=2.45e9)
        bad = optimize_s11(FREQS, notch_at(2.45e9, -5), target_freq=2.45e9)

        assert good < bad

    @pytest.mark.parametrize("mode", ["best", "min", "", "WORST"])
    def test_unknown_mode_raises(self, mode):
        with pytest.raises(ValueError, match="mode must be"):
            optimize_s11(FREQS, notch_at(2.45e9), freq_band=(1e9, 3e9), mode=mode)

    def test_missing_target_and_band_raises(self):
        with pytest.raises(ValueError, match="Provide target_freq or freq_band"):
            optimize_s11(FREQS, notch_at(2.45e9))


# ---------------------------------------------------------------------
# optimize_s21
# ---------------------------------------------------------------------
class TestOptimizeS21:
    """Note the sign asymmetry against ``optimize_s11``: both are costs to be
    minimised, but *more* transmission is better while *less* reflection is
    better, so ``optimize_s21`` negates the dB value. Flipping that turns an
    optimizer into a pessimizer, which is why it is pinned here."""

    def test_cost_is_the_negated_insertion_loss(self):
        s21 = s_from_db(np.full(len(FREQS), -1.5))

        assert optimize_s21(FREQS, s21, target_freq=2.45e9) == pytest.approx(1.5)

    def test_better_transmission_gives_a_lower_cost(self):
        lossy = optimize_s21(
            FREQS, s_from_db(np.full(len(FREQS), -10.0)), target_freq=2.45e9
        )
        good = optimize_s21(
            FREQS, s_from_db(np.full(len(FREQS), -0.5)), target_freq=2.45e9
        )

        assert good < lossy

    def test_worst_mode_takes_the_least_transmitting_point(self):
        """``worst`` is ``-min(S21_dB)`` -- the opposite reduction to the S11
        version, since here the worst case is the smallest value."""
        s21 = s_from_db(np.linspace(-1.0, -20.0, len(FREQS)))

        assert optimize_s21(
            FREQS, s21, freq_band=(1e9, 3e9), mode="worst"
        ) == pytest.approx(20.0, abs=0.1)

    def test_mean_mode_returns_the_negated_band_average(self):
        s21 = s_from_db(np.full(len(FREQS), -1.5))

        assert optimize_s21(
            FREQS, s21, freq_band=(1e9, 3e9), mode="mean"
        ) == pytest.approx(1.5, abs=1e-6)

    def test_target_freq_takes_precedence_over_freq_band(self):
        s21 = s_from_db(np.linspace(-1.0, -20.0, len(FREQS)))

        assert optimize_s21(
            FREQS, s21, target_freq=2.45e9, freq_band=(1e9, 3e9), mode="mean"
        ) == optimize_s21(FREQS, s21, target_freq=2.45e9)

    def test_missing_target_and_band_raises(self):
        with pytest.raises(ValueError, match="Provide target_freq or freq_band"):
            optimize_s21(FREQS, s_from_db(np.full(len(FREQS), -1.5)))

    @pytest.mark.parametrize("mode", ["best", "", "threshold"])
    def test_unknown_mode_raises(self, mode):
        """``threshold`` exists on ``optimize_s11`` but not here; asking for
        it must fail loudly rather than silently falling back."""
        with pytest.raises(ValueError, match="mode must be 'mean' or 'worst'"):
            optimize_s21(
                FREQS,
                s_from_db(np.full(len(FREQS), -1.5)),
                freq_band=(1e9, 3e9),
                mode=mode,
            )


# ---------------------------------------------------------------------
# param_sweep
# ---------------------------------------------------------------------
class TestParamSweep:
    @pytest.fixture
    def recorder(self):
        """A ``simulate_fn`` that records its calls and returns fake data."""
        calls = []

        def simulate(sweep_path, sweep, values):
            calls.append((Path(sweep_path), sweep, tuple(values)))
            return SimData(
                FREQS,
                notch_at(2.45e9),
                None,
                np.ones_like(FREQS) * 50j,
                np.ones_like(FREQS),
                1.0,
                np.ones_like(FREQS),
                np.ones_like(FREQS),
                50.0,
            )

        simulate.calls = calls
        return simulate

    def test_runs_once_per_value(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert len(recorder.calls) == 3

    def test_values_come_from_linspace(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert [values[0] for _p, _s, values in recorder.calls] == pytest.approx(
            [1.0, 2.0, 3.0]
        )

    def test_multiple_parameters_form_a_cartesian_product(self, recorder, tmp_path):
        param_sweep(recorder, {"w": (1.0, 2.0, 2), "l": (10.0, 30.0, 3)}, tmp_path)

        assert len(recorder.calls) == 6

    def test_product_is_ordered_with_the_last_key_varying_fastest(
        self, recorder, tmp_path
    ):
        param_sweep(recorder, {"w": (1.0, 2.0, 2), "l": (10.0, 20.0, 2)}, tmp_path)

        assert [values for _p, _s, values in recorder.calls] == [
            (1.0, 10.0),
            (1.0, 20.0),
            (2.0, 10.0),
            (2.0, 20.0),
        ]

    def test_each_run_gets_its_own_directory(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        paths = [path for path, _s, _v in recorder.calls]

        assert len(set(paths)) == 3
        assert all(path.is_dir() for path in paths)

    def test_directories_are_named_after_the_parameter_values(self, recorder, tmp_path):
        """The directory name is how a user finds the result for a given
        parameter combination afterwards."""
        param_sweep(recorder, {"width": (1.0, 2.0, 2)}, tmp_path)

        names = sorted(path.name for path, _s, _v in recorder.calls)

        assert names == ["width_1.0", "width_2.0"]

    def test_directories_live_under_a_sweep_subfolder(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 2.0, 2)}, tmp_path)

        for path, _s, _v in recorder.calls:
            assert path.parent == tmp_path / "sweep"

    def test_sweep_flag_is_passed_through(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 2.0, 2)}, tmp_path, sweep=False)

        assert all(sweep is False for _p, sweep, _v in recorder.calls)

    def test_returns_the_collected_results(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert len(results) == 3
        assert all(isinstance(item, SimData) for item in results)

    def test_single_point_sweep(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (2.0, 2.0, 1)}, tmp_path)

        assert len(results) == 1

    def test_empty_sweep_definition_runs_nothing(self, recorder, tmp_path):
        """``product()`` of nothing yields one empty tuple, so this documents
        which of the two plausible behaviours actually happens."""
        results = param_sweep(recorder, {}, tmp_path)

        assert len(recorder.calls) == len(results)


# ---------------------------------------------------------------------
# optimize_s_params
# ---------------------------------------------------------------------
class TestOptimizeSParams:
    def test_minimises_a_simple_quadratic(self, tmp_path, capsys):
        """Nelder-Mead on a bowl with its minimum at 3.0; the optimizer must
        get there and print the result."""

        def simulate(output_path, optimize, optimize_val):
            return float((optimize_val[0] - 3.0) ** 2)

        optimize_s_params(simulate, {"width": 0.5}, tmp_path)

        out = capsys.readouterr().out
        assert "optimal width" in out
        value = float(out.split("optimal width =")[1].split()[0])
        assert value == pytest.approx(3.0, abs=1e-2)

    def test_passes_the_output_path_and_optimize_flag(self, tmp_path):
        seen = []

        def simulate(output_path, optimize, optimize_val):
            seen.append((output_path, optimize))
            return float(optimize_val[0] ** 2)

        optimize_s_params(simulate, {"width": 1.0}, tmp_path)

        assert all(path == tmp_path for path, _flag in seen)
        assert all(flag is True for _path, flag in seen)

    def test_parameter_vector_follows_the_dict_order(self, tmp_path):
        seen = []

        def simulate(output_path, optimize, optimize_val):
            seen.append(tuple(optimize_val))
            return float(sum((v - i) ** 2 for i, v in enumerate(optimize_val)))

        optimize_s_params(simulate, {"a": 1.0, "b": 2.0, "c": 3.0}, tmp_path)

        assert seen[0] == pytest.approx((1.0, 2.0, 3.0))

    def test_starts_from_x0(self, tmp_path):
        seen = []

        def simulate(output_path, optimize, optimize_val):
            seen.append(float(optimize_val[0]))
            return float((optimize_val[0] - 3.0) ** 2)

        optimize_s_params(simulate, {"width": 7.25}, tmp_path)

        assert seen[0] == pytest.approx(7.25)

    def test_bounds_are_respected(self, tmp_path, capsys):
        """The unconstrained minimum is 3.0; bounding at 2.0 must stop there."""

        def simulate(output_path, optimize, optimize_val):
            return float((optimize_val[0] - 3.0) ** 2)

        optimize_s_params(simulate, {"width": 0.5}, tmp_path, bounds=[(0.0, 2.0)])

        out = capsys.readouterr().out
        value = float(out.split("optimal width =")[1].split()[0])
        assert value <= 2.0 + 1e-6

    def test_stalled_optimization_exits_quietly(self, tmp_path):
        """A flat cost surface stalls the callback, which raises StopIteration
        internally. That must not escape to the caller."""

        def simulate(output_path, optimize, optimize_val):
            return 1.0

        optimize_s_params(simulate, {"width": 1.0}, tmp_path)

    def test_returns_none(self, tmp_path):
        def simulate(output_path, optimize, optimize_val):
            return float(optimize_val[0] ** 2)

        assert optimize_s_params(simulate, {"width": 1.0}, tmp_path) is None
