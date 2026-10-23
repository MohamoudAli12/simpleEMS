"""Tests for the objective functions, sweep, and optimizer in ``sim_tools``.

``minimize_s_param``/``maximize_s_param`` are pure functions of an S-parameter array,
so they are tested directly against hand-computed dB values. ``param_sweep``
and ``optimize_s_params`` take a user-supplied ``simulate_fn``, which makes
them testable with a cheap analytic stand-in instead of a solver -- the thing
under test is the orchestration (how many calls, with what arguments, in what
order), not the electromagnetics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import (  # noqa: E402
    SimData,
    SweepData,
    maximize_s_param,
    minimize_s_param,
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
# minimize_s_param
# ---------------------------------------------------------------------
class TestMinimizeSParam:
    def test_target_freq_returns_the_value_at_that_point(self):
        s11 = notch_at(2.45e9)
        index = np.abs(FREQS - 2.45e9).argmin()
        expected = 20 * np.log10(np.abs(s11[index]))

        assert minimize_s_param(FREQS, s11, target_freq=2.45e9) == pytest.approx(
            expected
        )

    def test_target_freq_snaps_to_the_nearest_grid_point(self):
        """The requested frequency need not be on the grid."""
        s11 = notch_at(2.45e9)

        exact = minimize_s_param(FREQS, s11, target_freq=2.45e9)
        offset = minimize_s_param(FREQS, s11, target_freq=2.4500001e9)

        assert exact == offset

    def test_target_freq_takes_precedence_over_freq_band(self):
        """Documented: when both are given, ``freq_band``, ``mode`` and
        ``threshold`` are ignored."""
        s11 = notch_at(2.45e9)

        with_band = minimize_s_param(
            FREQS, s11, target_freq=2.45e9, freq_band=(1e9, 3e9), mode="mean"
        )
        without = minimize_s_param(FREQS, s11, target_freq=2.45e9)

        assert with_band == without

    def test_worst_mode_returns_the_maximum_in_band(self):
        s11 = s_from_db(np.linspace(-30, -3, len(FREQS)))

        assert minimize_s_param(
            FREQS, s11, freq_band=(1e9, 3e9), mode="worst"
        ) == pytest.approx(-3, abs=0.1)

    def test_mean_mode_returns_the_band_average(self):
        s11 = s_from_db(np.full(len(FREQS), -20.0))

        assert minimize_s_param(
            FREQS, s11, freq_band=(1e9, 3e9), mode="mean"
        ) == pytest.approx(-20.0, abs=1e-9)

    def test_threshold_mode_is_zero_when_fully_below_threshold(self):
        """Nothing exceeds -15 dB, so there is no penalty to pay."""
        s11 = s_from_db(np.full(len(FREQS), -30.0))

        assert minimize_s_param(
            FREQS, s11, freq_band=(1e9, 3e9), mode="threshold", threshold=-15
        ) == pytest.approx(0.0)

    def test_threshold_mode_accumulates_the_excess(self):
        s11 = s_from_db(np.full(len(FREQS), -10.0))
        in_band = np.sum((FREQS >= 1e9) & (FREQS <= 3e9))

        penalty = minimize_s_param(
            FREQS, s11, freq_band=(1e9, 3e9), mode="threshold", threshold=-15
        )

        assert penalty == pytest.approx(5.0 * in_band, rel=1e-6)

    def test_band_restricts_the_evaluation(self):
        """A bad match outside the band of interest must not be counted."""
        s11 = s_from_db(np.where(FREQS < 2e9, -3.0, -30.0))

        assert minimize_s_param(
            FREQS, s11, freq_band=(2.5e9, 3e9), mode="worst"
        ) == pytest.approx(-30.0, abs=0.1)

    def test_lower_cost_means_a_better_match(self):
        """The optimizer minimises this, so the ordering has to be right."""
        good = minimize_s_param(FREQS, notch_at(2.45e9, -30), target_freq=2.45e9)
        bad = minimize_s_param(FREQS, notch_at(2.45e9, -5), target_freq=2.45e9)

        assert good < bad

    @pytest.mark.parametrize("mode", ["best", "min", "", "WORST"])
    def test_unknown_mode_raises(self, mode):
        with pytest.raises(ValueError, match="mode must be"):
            minimize_s_param(FREQS, notch_at(2.45e9), freq_band=(1e9, 3e9), mode=mode)

    def test_missing_target_and_band_raises(self):
        with pytest.raises(ValueError, match="Provide target_freq or freq_band"):
            minimize_s_param(FREQS, notch_at(2.45e9))


# ---------------------------------------------------------------------
# maximize_s_param
# ---------------------------------------------------------------------
class TestMaximizeSParam:
    """Note the sign asymmetry against ``minimize_s_param``: both are costs
    to be minimised, but here a *higher* S-parameter is better, so
    ``maximize_s_param`` negates the dB value. Flipping that turns an
    optimizer into a pessimizer, which is why it is pinned here."""

    def test_cost_is_the_negated_insertion_loss(self):
        s21 = s_from_db(np.full(len(FREQS), -1.5))

        assert maximize_s_param(FREQS, s21, target_freq=2.45e9) == pytest.approx(1.5)

    def test_better_transmission_gives_a_lower_cost(self):
        lossy = maximize_s_param(
            FREQS, s_from_db(np.full(len(FREQS), -10.0)), target_freq=2.45e9
        )
        good = maximize_s_param(
            FREQS, s_from_db(np.full(len(FREQS), -0.5)), target_freq=2.45e9
        )

        assert good < lossy

    def test_worst_mode_takes_the_least_transmitting_point(self):
        """``worst`` is ``-min(S21_dB)`` -- the opposite reduction to the S11
        function, since here the worst case is the smallest value."""
        s21 = s_from_db(np.linspace(-1.0, -20.0, len(FREQS)))

        assert maximize_s_param(
            FREQS, s21, freq_band=(1e9, 3e9), mode="worst"
        ) == pytest.approx(20.0, abs=0.1)

    def test_mean_mode_returns_the_negated_band_average(self):
        s21 = s_from_db(np.full(len(FREQS), -1.5))

        assert maximize_s_param(
            FREQS, s21, freq_band=(1e9, 3e9), mode="mean"
        ) == pytest.approx(1.5, abs=1e-6)

    def test_target_freq_takes_precedence_over_freq_band(self):
        s21 = s_from_db(np.linspace(-1.0, -20.0, len(FREQS)))

        assert maximize_s_param(
            FREQS, s21, target_freq=2.45e9, freq_band=(1e9, 3e9), mode="mean"
        ) == maximize_s_param(FREQS, s21, target_freq=2.45e9)

    def test_missing_target_and_band_raises(self):
        with pytest.raises(ValueError, match="Provide target_freq or freq_band"):
            maximize_s_param(FREQS, s_from_db(np.full(len(FREQS), -1.5)))

    def test_threshold_mode_is_zero_when_fully_above_threshold(self):
        """Nothing drops below -1 dB, so there is no shortfall to pay."""
        s21 = s_from_db(np.full(len(FREQS), -0.5))

        assert maximize_s_param(
            FREQS, s21, freq_band=(1e9, 3e9), mode="threshold", threshold=-1
        ) == pytest.approx(0.0)

    def test_threshold_mode_accumulates_the_shortfall(self):
        s21 = s_from_db(np.full(len(FREQS), -4.0))
        in_band = np.sum((FREQS >= 1e9) & (FREQS <= 3e9))

        shortfall = maximize_s_param(
            FREQS, s21, freq_band=(1e9, 3e9), mode="threshold", threshold=-1
        )

        assert shortfall == pytest.approx(3.0 * in_band, rel=1e-6)

    @pytest.mark.parametrize("mode", ["best", "min", "", "WORST"])
    def test_unknown_mode_raises(self, mode):
        with pytest.raises(ValueError, match="mode must be"):
            maximize_s_param(
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
                None,
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

    def test_returns_sweep_data(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert isinstance(results, SweepData)

    def test_stacks_one_row_per_run(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert results.s11.shape == (3, len(FREQS))
        assert results.z11.shape == (3, len(FREQS))
        assert results.vswr.shape == (3, len(FREQS))
        assert results.input_power.shape == (3,)

    def test_freqs_are_shared_not_stacked(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert results.freqs == pytest.approx(FREQS)

    def test_missing_ports_stay_none(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert results.s21 is None
        assert results.s31 is None
        assert results.s41 is None

    def test_rows_follow_the_run_order(self, tmp_path):
        """Row ``i`` must be the result of run ``i``, or the labels lie."""

        def simulate(sweep_path, sweep, values):
            s11 = np.full_like(FREQS, values[0], dtype=complex)
            return SimData(
                FREQS, s11, None, None, None, s11, s11.real, 1.0, s11, s11, 50.0
            )

        results = param_sweep(simulate, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert results.s11[:, 0].real == pytest.approx([1.0, 2.0, 3.0])

    def test_labels_name_each_run(self, recorder, tmp_path):
        results = param_sweep(
            recorder, {"w": (1.0, 2.0, 2), "l": (10.0, 20.0, 2)}, tmp_path
        )

        assert results.labels == [
            "w=1.0, l=10.0",
            "w=1.0, l=20.0",
            "w=2.0, l=10.0",
            "w=2.0, l=20.0",
        ]

    def test_values_hold_the_swept_parameters(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (1.0, 2.0, 2)}, tmp_path)

        assert results.values == [{"width": 1.0}, {"width": 2.0}]

    def test_plots_nothing(self, recorder, tmp_path):
        param_sweep(recorder, {"width": (1.0, 3.0, 3)}, tmp_path)

        assert plt.get_fignums() == []

    def test_single_point_sweep(self, recorder, tmp_path):
        results = param_sweep(recorder, {"width": (2.0, 2.0, 1)}, tmp_path)

        assert results.s11.shape == (1, len(FREQS))

    def test_empty_sweep_is_an_error(self, recorder, tmp_path):
        with pytest.raises(ValueError, match="no parameter combinations"):
            param_sweep(recorder, {"width": (1.0, 3.0, 0)}, tmp_path)

    def test_runs_with_different_frequencies_are_an_error(self, tmp_path):
        def simulate(sweep_path, sweep, values):
            freqs = FREQS * values[0]
            s11 = np.ones_like(freqs, dtype=complex)
            return SimData(
                freqs, s11, None, None, None, s11, s11.real, 1.0, s11, s11, 50.0
            )

        with pytest.raises(ValueError, match="different frequency points"):
            param_sweep(simulate, {"width": (1.0, 2.0, 2)}, tmp_path)

    def test_empty_sweep_definition_runs_nothing(self, recorder, tmp_path):
        """``product()`` of nothing yields one empty tuple, so this documents
        which of the two plausible behaviours actually happens."""
        results = param_sweep(recorder, {}, tmp_path)

        assert len(recorder.calls) == len(results.labels)


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

    def test_a_simulation_that_stops_the_run_exits_quietly(self, tmp_path, capsys):
        """SciPy only catches StopIteration from the callback; one raised by
        the simulation itself reaches optimize_s_params, which ends the run
        without printing a result."""

        def simulate(output_path, optimize, optimize_val):
            raise StopIteration

        optimize_s_params(simulate, {"width": 1.0}, tmp_path)

        assert "optimal width" not in capsys.readouterr().out

    def test_returns_none(self, tmp_path):
        def simulate(output_path, optimize, optimize_val):
            return float(optimize_val[0] ** 2)

        assert optimize_s_params(simulate, {"width": 1.0}, tmp_path) is None

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        ("algorithm", "bounds"),
        [
            ("Nelder-Mead", [(0.0, 5.0)]),
            ("Powell", [(0.0, 5.0)]),
            ("COBYLA", [(0.0, 5.0)]),
            ("COBYQA", [(0.0, 5.0)]),
            ("L-BFGS-B", [(0.0, 5.0)]),
            ("TNC", [(0.0, 5.0)]),
            ("SLSQP", [(0.0, 5.0)]),
            ("trust-constr", [(0.0, 5.0)]),
            ("CG", None),
            ("BFGS", None),
        ],
        ids=[
            "nelder-mead",
            "powell",
            "cobyla",
            "cobyqa",
            "l-bfgs-b",
            "tnc",
            "slsqp",
            "trust-constr",
            "cg",
            "bfgs",
        ],
    )
    def test_every_supported_algorithm_finds_the_minimum(
        self, tmp_path, capsys, algorithm, bounds
    ):
        """Each supported method must reach the bowl's minimum at 3.0 without
        a SciPy warning (unknown options, ignored bounds, deprecated options)."""

        def simulate(output_path, optimize, optimize_val):
            return float((optimize_val[0] - 3.0) ** 2)

        optimize_s_params(
            simulate, {"width": 0.5}, tmp_path, bounds=bounds, algorithm=algorithm
        )

        out = capsys.readouterr().out
        value = float(out.split("optimal width =")[1].split()[0])
        assert value == pytest.approx(3.0, abs=1e-2)

    @pytest.mark.parametrize(
        "algorithm",
        ["Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov", "foo"],
        ids=["newton-cg", "dogleg", "trust-ncg", "trust-exact", "trust-krylov", "foo"],
    )
    def test_rejects_an_unsupported_algorithm(self, tmp_path, algorithm):
        """Methods that need an analytic Jacobian, and unknown names, fail
        before any simulation runs."""
        calls = []

        def simulate(output_path, optimize, optimize_val):
            calls.append(optimize_val)
            return 0.0

        with pytest.raises(ValueError, match="algorithm must be one of"):
            optimize_s_params(simulate, {"width": 1.0}, tmp_path, algorithm=algorithm)

        assert calls == []

    @pytest.mark.parametrize("algorithm", ["CG", "BFGS"], ids=["cg", "bfgs"])
    def test_rejects_bounds_for_an_unbounded_algorithm(self, tmp_path, algorithm):
        def simulate(output_path, optimize, optimize_val):
            return 0.0

        with pytest.raises(ValueError, match="does not support bounds"):
            optimize_s_params(
                simulate,
                {"width": 1.0},
                tmp_path,
                bounds=[(0.0, 2.0)],
                algorithm=algorithm,
            )
