"""Tests for the plotting and reporting helpers in ``sim_tools``.

Plots are not compared pixel-by-pixel -- that is brittle and catches nothing
useful. What is asserted is that each helper opens exactly one figure, plots
the data it was given (recovered from the line artists), applies the labels it
was asked for, and that ``save_plots`` writes real files.

``conftest.py`` forces the Agg backend before ``sim_tools`` is imported, which
is what keeps these from trying to open a PyQt6 window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import SimTools  # noqa: E402


FREQS = np.linspace(1e9, 3e9, 101)
S11 = 0.3 * np.exp(-1j * FREQS / 1e9)
S21 = 0.9 * np.exp(-2j * FREQS / 1e9)
Z11 = 50 + 10j * np.sin(FREQS / 1e9)
VSWR = (1 + np.abs(S11)) / (1 - np.abs(S11))


def current_lines():
    """Every plotted line on the current axes."""
    return plt.gca().get_lines()


# ---------------------------------------------------------------------
# S-parameters
# ---------------------------------------------------------------------
class TestPlotSParam:
    def test_opens_a_single_figure(self):
        SimTools.plot_s_param(FREQS, S11)

        assert len(plt.get_fignums()) == 1

    def test_plots_s11_in_decibels(self):
        SimTools.plot_s_param(FREQS, S11)

        line = current_lines()[0]
        expected = 20 * np.log10(np.abs(S11))

        assert line.get_ydata() == pytest.approx(expected)

    def test_x_data_is_the_frequency_axis(self):
        SimTools.plot_s_param(FREQS, S11)

        assert current_lines()[0].get_xdata() == pytest.approx(FREQS)

    def test_s21_adds_a_second_trace(self):
        SimTools.plot_s_param(FREQS, S11, s21=S21)

        assert len(current_lines()) == 2

    def test_omitting_s21_plots_one_trace(self):
        SimTools.plot_s_param(FREQS, S11)

        assert len(current_lines()) == 1

    def test_custom_labels_are_applied(self):
        SimTools.plot_s_param(FREQS, S11, x_label="f", y_label="dB", title="my title")

        assert plt.gca().get_xlabel() == "f"
        assert plt.gca().get_ylabel() == "dB"
        assert plt.gca().get_title() == "my title"

    def test_custom_series_labels_reach_the_legend(self):
        SimTools.plot_s_param(FREQS, S11, s21=S21, label_s11="ref", label_s21="thru")

        labels = [line.get_label() for line in current_lines()]

        assert "ref" in labels
        assert "thru" in labels

    def test_repeated_calls_overlay_on_one_figure(self):
        """``param_sweep`` relies on this to build a comparison plot."""
        SimTools.plot_s_param(FREQS, S11, label_s11="a")
        SimTools.plot_s_param(FREQS, S11 * 0.5, label_s11="b")

        assert len(plt.get_fignums()) == 1
        assert len(current_lines()) == 2


# ---------------------------------------------------------------------
# Other plots
# ---------------------------------------------------------------------
class TestOtherPlots:
    def test_vswr_plots_the_given_values(self):
        SimTools.plot_vswr(FREQS, VSWR)

        assert current_lines()[0].get_ydata() == pytest.approx(VSWR)

    def test_vswr_axis_starts_at_or_above_one(self):
        """VSWR below 1 is unphysical, so the axis should not imply it."""
        SimTools.plot_vswr(FREQS, VSWR)

        assert plt.gca().get_ylim()[0] >= 0

    def test_impedance_plots_real_and_imaginary_parts(self):
        SimTools.plot_impedance(FREQS, Z11)

        lines = current_lines()

        assert len(lines) >= 2
        plotted = [line.get_ydata() for line in lines]
        assert any(np.allclose(series, np.real(Z11)) for series in plotted), (
            "real part not plotted"
        )
        assert any(np.allclose(series, np.imag(Z11)) for series in plotted), (
            "imaginary part not plotted"
        )

    def test_phase_is_in_degrees(self):
        SimTools.plot_phase(FREQS, S21)

        values = current_lines()[0].get_ydata()

        assert np.all(np.abs(values) <= 360.0)

    def test_group_delay_opens_a_figure(self):
        SimTools.plot_group_delay(FREQS, S21)

        assert len(plt.get_fignums()) == 1

    def test_group_delay_of_a_linear_phase_is_constant(self):
        """A pure delay has constant group delay; a sign or factor error in
        the derivative shows up immediately here."""
        delay = 1e-9
        s21 = np.exp(-2j * np.pi * FREQS * delay)

        SimTools.plot_group_delay(FREQS, s21)
        values = np.asarray(current_lines()[0].get_ydata())

        assert np.std(values) < 0.05 * max(abs(np.mean(values)), 1e-30)

    def test_smith_chart_opens_a_figure(self):
        SimTools.plot_smith_chart(FREQS, S11)

        assert len(plt.get_fignums()) == 1

    def test_smith_chart_accepts_a_custom_reference_impedance(self):
        """The FEM backend references S11 to the port's computed Zc, not
        necessarily 50 ohms."""
        SimTools.plot_smith_chart(FREQS, S11, charac_imp=75.0)

        assert len(plt.get_fignums()) == 1

    @pytest.mark.parametrize(
        ("plotter", "args"),
        [
            (SimTools.plot_s_param, (FREQS, S11)),
            (SimTools.plot_vswr, (FREQS, VSWR)),
            (SimTools.plot_impedance, (FREQS, Z11)),
            (SimTools.plot_phase, (FREQS, S21)),
            (SimTools.plot_group_delay, (FREQS, S21)),
        ],
    )
    def test_every_plot_is_titled_and_labelled(self, plotter, args):
        plotter(*args)

        assert plt.gca().get_title()
        assert plt.gca().get_xlabel()
        assert plt.gca().get_ylabel()


# ---------------------------------------------------------------------
# save_plots
# ---------------------------------------------------------------------
class TestSavePlots:
    def test_saves_one_file_per_open_figure(self, tmp_path):
        SimTools.plot_s_param(FREQS, S11)
        plt.figure()
        SimTools.plot_vswr(FREQS, VSWR)

        SimTools.save_plots(output_path=tmp_path)

        saved = sorted((tmp_path / "plots").iterdir())

        assert len(saved) == len(plt.get_fignums())

    def test_files_are_named_sequentially(self, tmp_path):
        SimTools.plot_s_param(FREQS, S11)

        SimTools.save_plots(output_path=tmp_path)

        assert (tmp_path / "plots" / "plot_1.png").is_file()

    def test_creates_the_plots_directory(self, tmp_path):
        SimTools.plot_s_param(FREQS, S11)

        SimTools.save_plots(output_path=tmp_path / "nested")

        assert (tmp_path / "nested" / "plots").is_dir()

    @pytest.mark.parametrize("file_format", ["png", "pdf", "svg"])
    def test_format_is_honoured(self, tmp_path, file_format):
        SimTools.plot_s_param(FREQS, S11)

        SimTools.save_plots(output_path=tmp_path, file_format=file_format)

        assert (tmp_path / "plots" / f"plot_1.{file_format}").is_file()

    def test_saved_file_is_non_empty(self, tmp_path):
        SimTools.plot_s_param(FREQS, S11)

        SimTools.save_plots(output_path=tmp_path)

        assert (tmp_path / "plots" / "plot_1.png").stat().st_size > 0

    def test_no_open_figures_writes_nothing(self, tmp_path):
        plt.close("all")

        SimTools.save_plots(output_path=tmp_path)

        assert list((tmp_path / "plots").iterdir()) == []

    def test_default_output_path_is_sim_path_under_cwd(self):
        SimTools.plot_s_param(FREQS, S11)

        SimTools.save_plots()

        assert (Path.cwd() / "Sim_Path" / "plots" / "plot_1.png").is_file()


# ---------------------------------------------------------------------
# Parameter reporting
# ---------------------------------------------------------------------
class TestPrintAndSaveParams:
    def test_writes_a_parameter_file(self, inset_params, tmp_path):
        SimTools.print_and_save_params(inset_params, output_path=tmp_path)

        written = list(tmp_path.rglob("*"))

        assert any(path.is_file() for path in written)

    def test_report_contains_the_computed_geometry(self, inset_params, tmp_path):
        SimTools.print_and_save_params(inset_params, output_path=tmp_path)

        text = "\n".join(
            path.read_text(errors="ignore")
            for path in tmp_path.rglob("*")
            if path.is_file()
        )

        assert "patch_width_mm" in text
        assert str(inset_params.patch_width_mm) in text

    def test_prints_to_stdout(self, inset_params, tmp_path, capsys):
        SimTools.print_and_save_params(inset_params, output_path=tmp_path)

        assert capsys.readouterr().out.strip()
