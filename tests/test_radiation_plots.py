"""Tests for the radiation, 3D-visualisation and orchestration parts of SimTools.

``test_plotting.py`` covers the 2D S-parameter family. This file covers the
rest: the polar radiation cuts, the PyVista surfaces, the field dump, and the
export wrappers.

Everything here is driven from a synthetic far field rather than a solve. The
plots take an ``nf2ff``-shaped object and never look at where it came from, so
a stub with a known analytic pattern exercises the same code an antenna would
and lets the tests assert on numbers that can be worked out by hand.

Two things are always stubbed: ``BackgroundPlotter``, which opens a Qt window
per 3D plot, and ``show_plots``, which blocks until those windows are closed.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

import pyvista as pv  # noqa: E402
from openEMS.nf2ff import nf2ff as OpenEMSNF2FF  # noqa: E402

from simpleEMS import sim_tools  # noqa: E402
from simpleEMS.sim_tools import DumpType, SimData, SimTools  # noqa: E402


class FakeFarField:
    """The subset of an ``nf2ff`` result the plots actually read."""

    def __init__(self, E_norm, Dmax, Prad, P_rad, theta, phi, Ploss=None):
        self.E_norm = E_norm
        self.Dmax = Dmax
        self.Prad = Prad
        self.P_rad = P_rad
        self.theta = theta
        self.phi = phi
        if Ploss is not None:
            self.Ploss = Ploss


class FakeNF2FF:
    """A far-field source with a known, closed-form broadside pattern.

    ``E_norm`` is ``cos(theta / 2) ** 2``: peak at theta = 0, independent of
    phi, and crossing -3 dB at theta = +-65.4 degrees, so the half-power
    beamwidth the directivity plot computes is a number this file can check
    against arithmetic rather than against itself.
    """

    def __init__(self, dmax=4.0, prad=0.75, ploss=0.25, with_ploss=True):
        self.dmax = dmax
        self.prad = prad
        self.ploss = ploss
        self.with_ploss = with_ploss
        self.calls = []

    def CalcNF2FF(  # noqa: N802 - matches the openEMS method name
        self,
        output_path,
        freq,
        theta,
        phi,
        read_cached=False,
        outfile=None,
        verbose=0,
    ):
        self.calls.append(
            {
                "output_path": output_path,
                "freq": freq,
                "theta": np.atleast_1d(theta),
                "phi": np.atleast_1d(phi),
                "read_cached": read_cached,
                "outfile": outfile,
            }
        )
        theta_deg = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_deg = np.atleast_1d(np.asarray(phi, dtype=float))
        th = np.deg2rad(theta_deg)[:, None] * np.ones((1, phi_deg.size))
        e_norm = np.cos(th / 2.0) ** 2
        return FakeFarField(
            E_norm=e_norm,
            Dmax=np.array([self.dmax]),
            Prad=np.array([self.prad]),
            P_rad=e_norm**2,
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
            Ploss=np.array([self.ploss]) if self.with_ploss else None,
        )

    def calc_power(self, output_path, freq, verbose=True):
        """The FEM adapter's pattern-free power solve."""
        self.calls.append({"output_path": output_path, "freq": freq})
        return self.prad, self.ploss


def make_sim_data(freqs, input_power, s11=None):
    """A ``SimData`` carrying only what ``compute_nf2ff_data`` reads. S11
    defaults to a perfect match, so the mismatch efficiency is 1."""
    zeros = np.zeros(len(freqs), dtype=complex)
    return SimData(
        freqs=np.asarray(freqs, dtype=float),
        s11=zeros if s11 is None else np.asarray(s11, dtype=complex),
        s21=None,
        s31=None,
        s41=None,
        z11=zeros,
        vswr=np.ones(len(freqs)),
        input_power=np.asarray(input_power, dtype=float),
        port_voltage=zeros,
        port_current=zeros,
    )


@pytest.fixture
def nf2ff():
    return FakeNF2FF()


@pytest.fixture
def sim_data():
    """1 W accepted across the band."""
    return make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 1.0, 1.0])


@pytest.fixture
def nf2ff_data(nf2ff, sim_data, tmp_path):
    """What ``compute_nf2ff_data`` hands to every radiation plot."""
    return SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)


@pytest.fixture
def no_windows(monkeypatch):
    """Replace the Qt-backed 3D plotter with a recorder.

    Each 3D plot opens a ``BackgroundPlotter``; unstubbed, a headless run
    either fails outright or leaks a window per test.
    """
    opened = []

    class Recorder:
        def __init__(self, title=None, **kwargs):
            self.title = title
            self.meshes = []
            self.texts = []
            opened.append(self)

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def add_text(self, text, **kwargs):
            self.texts.append(text)

    monkeypatch.setattr(sim_tools, "BackgroundPlotter", Recorder)
    return opened


@pytest.fixture
def figures():
    """Open matplotlib figures, by number, after the test body runs."""
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------
# compute_nf2ff_data
# ---------------------------------------------------------------------
class TestComputeNF2FFData:
    def test_solves_the_sphere_and_the_three_principal_cuts(
        self, nf2ff, sim_data, tmp_path
    ):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)

        sphere, xz_cut, yz_cut, xy_cut = nf2ff.calls
        assert sphere["theta"][0] == 0 and sphere["theta"][-1] == 180
        assert sphere["phi"][0] == 0 and sphere["phi"][-1] == 360
        # xz cut: theta swept at phi = 0
        assert xz_cut["phi"].tolist() == [0]
        # yz cut: theta swept at phi = 90
        assert yz_cut["phi"].tolist() == [90]
        # xy cut: phi swept at theta = 90
        assert xy_cut["theta"].tolist() == [90]

    def test_the_sphere_grid_is_two_degree_steps(self, nf2ff, sim_data, tmp_path):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)

        sphere = nf2ff.calls[0]
        assert sphere["theta"].size == 91
        assert sphere["phi"].size == 181

    @pytest.mark.parametrize(
        ("call_index", "swept"),
        [(1, "theta"), (2, "theta"), (3, "phi")],
        ids=["xz", "yz", "xy"],
    )
    def test_each_cut_sweeps_a_full_circle_in_fine_steps(
        self, nf2ff, sim_data, tmp_path, call_index, swept
    ):
        """The beamwidth is read off the sampled curve, so the cut has to be
        fine enough to place the -3 dB crossing accurately."""
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)

        angles = nf2ff.calls[call_index][swept]
        assert angles.min() == -180.0
        assert angles.max() == 180.0
        assert np.diff(angles).max() == pytest.approx(0.1)

    def test_each_solve_writes_its_own_file(self, nf2ff, sim_data, tmp_path):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)

        outfiles = [call["outfile"] for call in nf2ff.calls]
        assert outfiles == [
            "nf2ff_3d.h5",
            "nf2ff_xz.h5",
            "nf2ff_yz.h5",
            "nf2ff_xy.h5",
        ]

    def test_carries_the_frequency(self, nf2ff_data):
        assert nf2ff_data.freq == 2.45e9

    def test_read_cached_is_passed_through(self, nf2ff, sim_data, tmp_path):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path, read_cached=True)

        assert all(call["read_cached"] for call in nf2ff.calls)

    def test_the_output_path_is_passed_through(self, nf2ff, sim_data, tmp_path):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data, tmp_path)

        assert all(call["output_path"] == tmp_path for call in nf2ff.calls)

    def test_default_output_path_is_sim_path_under_cwd(self, nf2ff, sim_data):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data)

        assert nf2ff.calls[0]["output_path"] == Path.cwd() / "Sim_Path"

    def test_an_array_of_frequencies_is_rejected(self, nf2ff, sim_data, tmp_path):
        """One far field is one frequency; an array would silently use the
        first."""
        with pytest.raises(TypeError, match="only one frequency"):
            SimTools.compute_nf2ff_data(
                nf2ff, np.array([2.4e9, 2.5e9]), sim_data, tmp_path
            )

    def test_fdtd_efficiency_takes_the_accepted_power_at_freq(self, tmp_path):
        """openEMS reports no Ploss, so the accepted power comes from the port:
        the 2 W at 2.45 GHz, not the band's largest 4 W."""
        source = FakeNF2FF(prad=0.5, with_ploss=False)
        band = make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 2.0, 4.0])

        result = SimTools.compute_nf2ff_data(source, 2.45e9, band, tmp_path)

        assert result.efficiency == pytest.approx(0.25)

    def test_fem_efficiency_comes_from_ploss(self, tmp_path):
        """With Prad = 0.75 and Ploss = 0.25 the efficiency is 75%, whatever
        the port reports."""
        band = make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 2.0, 4.0])

        result = SimTools.compute_nf2ff_data(FakeNF2FF(), 2.45e9, band, tmp_path)

        assert result.efficiency == pytest.approx(0.75)

    def test_mismatch_efficiency_takes_s11_at_freq(self, tmp_path):
        """|S11| = 0.5 at 2.45 GHz reflects a quarter of the incident power,
        whatever S11 is elsewhere in the band."""
        band = make_sim_data(
            [2.4e9, 2.45e9, 2.5e9], [1.0, 1.0, 1.0], s11=[0.9, 0.5j, 0.9]
        )

        result = SimTools.compute_nf2ff_data(FakeNF2FF(), 2.45e9, band, tmp_path)

        assert result.mismatch_efficiency == pytest.approx(0.75)


# ---------------------------------------------------------------------
# plot_2d_rad_pattern
# ---------------------------------------------------------------------
class TestPlot2dRadPattern:
    def test_opens_one_figure_with_three_polar_axes(self, nf2ff_data, figures):
        """The xz, yz and xy cuts are drawn side by side in one figure."""
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        assert len(figures.get_fignums()) == 1
        assert len(figures.gcf().axes) == 3

    def test_every_axis_is_polar(self, nf2ff_data, figures):
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        assert all(ax.name == "polar" for ax in figures.gcf().axes)

    def test_the_pattern_is_normalised_to_its_peak(self, nf2ff_data, figures):
        """A polar cut is always plotted relative to its own maximum, so the
        peak sits at 0 dB whatever the absolute field level was."""
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(0.0)

    def test_the_broadside_peak_is_at_theta_zero(self, nf2ff_data, figures):
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        line = figures.gcf().axes[0].lines[0]
        peak_theta = np.rad2deg(line.get_xdata()[np.argmax(line.get_ydata())])
        assert peak_theta == pytest.approx(0.0)

    def test_each_cut_is_labelled(self, nf2ff_data, figures):
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        labels = [ax.lines[0].get_label() for ax in figures.gcf().axes]
        assert labels == ["xz-plane", "yz-plane", "xy-plane"]

    def test_the_radial_axis_spans_the_dynamic_range(self, nf2ff_data, figures):
        """The axis is anchored to the peak, not to the pattern minimum, so an
        omnidirectional cut draws as a circle instead of filling the plot."""
        SimTools.plot_2d_rad_pattern(nf2ff_data, dynamic_range_db=20)

        for ax in figures.gcf().axes:
            assert ax.get_ylim() == pytest.approx((-20.0, 0.0))

    def test_deep_nulls_clip_at_the_axis_floor(self, nf2ff_data, figures):
        """The fake pattern has a true null at theta = 180 degrees; it must
        land on the floor rather than wrap through the centre."""
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.min(ydata) == pytest.approx(-30.0)

    def test_the_frequency_is_in_the_title(self, nf2ff_data, figures):
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        assert "2.45" in figures.gcf()._suptitle.get_text()

    def test_zero_is_north_and_angles_run_clockwise(self, nf2ff_data, figures):
        """Antenna convention: theta = 0 points up the page."""
        SimTools.plot_2d_rad_pattern(nf2ff_data)

        for ax in figures.gcf().axes:
            assert ax.get_theta_offset() == pytest.approx(np.pi / 2)
            assert ax.get_theta_direction() == -1

    def test_solves_nothing_itself(self, nf2ff, nf2ff_data, figures):
        """The far field was solved once by compute_nf2ff_data; the plot only
        reads it."""
        calls_before = len(nf2ff.calls)

        SimTools.plot_2d_rad_pattern(nf2ff_data)

        assert len(nf2ff.calls) == calls_before


# ---------------------------------------------------------------------
# plot_2d_directivity
# ---------------------------------------------------------------------
class IsotropicNF2FF(FakeNF2FF):
    """A pattern that never drops 3 dB, so it has no beamwidth."""

    def CalcNF2FF(self, output_path, freq, theta, phi, **kwargs):  # noqa: N802
        theta_deg = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_deg = np.atleast_1d(np.asarray(phi, dtype=float))
        shape = (theta_deg.size, phi_deg.size)
        return FakeFarField(
            E_norm=np.ones(shape),
            Dmax=np.array([1.0]),
            Prad=np.array([1.0]),
            P_rad=np.ones(shape),
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
        )


class BackwardNF2FF(FakeNF2FF):
    """A backward lobe peaking at theta = 180 deg, the last sample of the
    sweep, so its -3 dB crossings lie on both sides of +-180 deg."""

    def CalcNF2FF(self, output_path, freq, theta, phi, **kwargs):  # noqa: N802
        theta_deg = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_deg = np.atleast_1d(np.asarray(phi, dtype=float))
        # The floor keeps the null at theta = 0 finite in dB.
        pattern = np.sin(np.deg2rad(theta_deg) / 2) ** 2 + 1e-6
        shape = (theta_deg.size, phi_deg.size)
        return FakeFarField(
            E_norm=np.broadcast_to(pattern[:, None], shape),
            Dmax=np.array([4.0]),
            Prad=np.array([1.0]),
            P_rad=np.ones(shape),
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
        )


class OffPlaneNF2FF(FakeNF2FF):
    """The broadside pattern scaled by ``(1 + sin(phi)^2) / 2``: full strength
    in the yz-plane but half of it in the xz-plane, so the xz cut misses the
    sphere's peak by 6 dB."""

    def CalcNF2FF(self, output_path, freq, theta, phi, **kwargs):  # noqa: N802
        theta_deg = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_deg = np.atleast_1d(np.asarray(phi, dtype=float))
        theta_part = np.cos(np.deg2rad(theta_deg) / 2.0)[:, None] ** 2
        phi_part = (1.0 + np.sin(np.deg2rad(phi_deg))[None, :] ** 2) / 2.0
        e_norm = theta_part * phi_part
        return FakeFarField(
            E_norm=e_norm,
            Dmax=np.array([4.0]),
            Prad=np.array([0.75]),
            P_rad=e_norm**2,
            theta=np.deg2rad(theta_deg),
            phi=np.deg2rad(phi_deg),
            Ploss=np.array([0.25]),
        )


def far_field_from(source, sim_data, tmp_path):
    return SimTools.compute_nf2ff_data(source, 2.45e9, sim_data, tmp_path)


def hpbw_annotation(figures):
    return next(
        t.get_text() for t in figures.gcf().axes[0].texts if "HPBW" in t.get_text()
    )


class TestPlot2dDirectivity:
    def test_opens_a_polar_and_a_cartesian_figure_by_default(self, nf2ff_data, figures):
        SimTools.plot_2d_directivity(nf2ff_data)

        figure_numbers = figures.get_fignums()
        assert len(figure_numbers) == 2
        assert figures.figure(figure_numbers[0]).axes[0].name == "polar"
        assert figures.figure(figure_numbers[1]).axes[0].name == "rectilinear"

    @pytest.mark.parametrize(
        ("style", "projection"),
        [("polar", "polar"), ("cartesian", "rectilinear")],
        ids=["polar", "cartesian"],
    )
    def test_a_single_style_opens_one_figure(
        self, nf2ff_data, figures, style, projection
    ):
        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        assert len(figures.get_fignums()) == 1
        assert figures.gcf().axes[0].name == projection

    def test_an_unknown_style_is_rejected(self, nf2ff_data):
        with pytest.raises(ValueError, match="style must be"):
            SimTools.plot_2d_directivity(nf2ff_data, style="smith")

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_the_peak_is_the_max_directivity(self, nf2ff_data, figures, style):
        """The curve is the pattern lifted by Dmax, so its peak is Dmax in
        dBi -- 10*log10(4) = 6.02 dBi here."""
        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(10 * np.log10(4.0), abs=1e-6)

    def test_a_cut_that_misses_the_main_lobe_sits_below_dmax(
        self, sim_data, tmp_path, figures
    ):
        """The cut is referenced to the sphere's peak, not its own: an xz
        field at half the sphere's peak sits 6 dB under Dmax, at 0 dBi."""
        nf2ff_data = far_field_from(OffPlaneNF2FF(), sim_data, tmp_path)

        SimTools.plot_2d_directivity(nf2ff_data, style="cartesian")

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        expected = 10 * np.log10(4.0) + 20 * np.log10(0.5)
        assert np.max(ydata) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_the_computed_beamwidth_matches_the_pattern(
        self, nf2ff_data, figures, style
    ):
        """cos^2(theta/2) falls to -3 dB at theta = +-65.4 deg, so the
        half-power beamwidth is about 131 deg."""
        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        value = float(hpbw_annotation(figures).split("=")[1].split("°")[0])
        assert value == pytest.approx(130.8, abs=1.0)

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_the_main_lobe_direction_is_reported(self, nf2ff_data, figures, style):
        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        annotation = hpbw_annotation(figures)
        direction = float(annotation.split("Main Lobe Direction =")[1].split("°")[0])
        assert direction == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_an_omnidirectional_cut_has_no_beamwidth(
        self, sim_data, tmp_path, figures, style
    ):
        """An isotropic pattern never drops 3 dB, so no beamwidth exists; the
        plot still reports the main lobe but leaves the HPBW out."""
        nf2ff_data = far_field_from(IsotropicNF2FF(), sim_data, tmp_path)

        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        texts = [t.get_text() for t in figures.gcf().axes[0].texts]
        assert any("Main Lobe Direction" in t for t in texts)
        assert not any("HPBW" in t for t in texts)

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_the_beamwidth_wraps_through_180_degrees(
        self, sim_data, tmp_path, figures, style
    ):
        """The backward lobe's beamwidth matches the broadside case of about
        131 deg, even though its -3 dB crossings straddle +-180 deg."""
        nf2ff_data = far_field_from(BackwardNF2FF(), sim_data, tmp_path)

        SimTools.plot_2d_directivity(nf2ff_data, style=style)

        value = float(hpbw_annotation(figures).split("=")[1].split("°")[0])
        assert value == pytest.approx(130.8, abs=1.0)

    def test_cartesian_beam_edges_fold_back_into_the_theta_range(
        self, sim_data, tmp_path, figures
    ):
        """A beam straddling +-180 deg has an edge past 180 deg; the cartesian
        view folds it back so both edge markers land on the plotted axis, one
        near each end."""
        nf2ff_data = far_field_from(BackwardNF2FF(), sim_data, tmp_path)

        SimTools.plot_2d_directivity(nf2ff_data, style="cartesian")

        ax = figures.gcf().axes[0]
        vertical_x = [
            line.get_xdata()[0]
            for line in ax.lines[1:]
            if line.get_xdata()[0] == line.get_xdata()[1]
        ]
        edges = sorted(x for x in vertical_x if abs(abs(x) - 180.0) > 1e-9)
        assert all(-180.0 <= x <= 180.0 for x in vertical_x)
        assert edges == pytest.approx([-114.6, 114.6], abs=1.0)

    def test_the_frequency_is_in_the_title(self, nf2ff_data, figures):
        SimTools.plot_2d_directivity(nf2ff_data, style="polar")

        assert "2.45" in figures.gcf()._suptitle.get_text()


# ---------------------------------------------------------------------
# plot_2d_gain
# ---------------------------------------------------------------------
class TestPlot2dGain:
    def test_opens_a_polar_and_a_cartesian_figure_by_default(self, nf2ff_data, figures):
        SimTools.plot_2d_gain(nf2ff_data)

        figure_numbers = figures.get_fignums()
        assert len(figure_numbers) == 2
        assert figures.figure(figure_numbers[0]).axes[0].name == "polar"
        assert figures.figure(figure_numbers[1]).axes[0].name == "rectilinear"

    @pytest.mark.parametrize(
        ("style", "projection"),
        [("polar", "polar"), ("cartesian", "rectilinear")],
        ids=["polar", "cartesian"],
    )
    def test_a_single_style_opens_one_figure(
        self, nf2ff_data, figures, style, projection
    ):
        SimTools.plot_2d_gain(nf2ff_data, style=style)

        assert len(figures.get_fignums()) == 1
        assert figures.gcf().axes[0].name == projection

    def test_an_unknown_style_is_rejected(self, nf2ff_data):
        with pytest.raises(ValueError, match="style must be"):
            SimTools.plot_2d_gain(nf2ff_data, style="smith")

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_fem_peak_is_dmax_scaled_by_the_efficiency(
        self, nf2ff_data, figures, style
    ):
        """Prad = 0.75 and Ploss = 0.25 give 75% efficiency, so the peak is
        10*log10(0.75 * 4) = 4.77 dBi."""
        SimTools.plot_2d_gain(nf2ff_data, style=style)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(10 * np.log10(3.0), abs=1e-6)

    def test_fdtd_efficiency_takes_the_accepted_power_at_freq(self, tmp_path, figures):
        """openEMS reports no Ploss: Prad = 0.5 W over the 2 W accepted at
        2.45 GHz is 25%, so the peak is 10*log10(0.25 * 4) = 0 dBi."""
        source = FakeNF2FF(prad=0.5, with_ploss=False)
        band = make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 2.0, 4.0])
        nf2ff_data = far_field_from(source, band, tmp_path)

        SimTools.plot_2d_gain(nf2ff_data, style="cartesian")

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(0.0, abs=1e-6)

    def test_the_peak_is_derated_by_the_port_mismatch(self, nf2ff, tmp_path, figures):
        """|S11| = 0.5 lets 75% of the incident power in, on top of the 75%
        radiation efficiency: 10*log10(4 * 0.75 * 0.75) = 3.52 dBi."""
        band = make_sim_data(
            [2.4e9, 2.45e9, 2.5e9], [1.0, 1.0, 1.0], s11=[0.5, 0.5, 0.5]
        )
        nf2ff_data = far_field_from(nf2ff, band, tmp_path)

        SimTools.plot_2d_gain(nf2ff_data, style="cartesian")

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(10 * np.log10(2.25), abs=1e-6)

    def test_a_cut_that_misses_the_main_lobe_sits_below_the_peak_gain(
        self, sim_data, tmp_path, figures
    ):
        """An xz field at half the sphere's peak sits 6 dB under the 4.77 dBi
        peak gain."""
        nf2ff_data = far_field_from(OffPlaneNF2FF(), sim_data, tmp_path)

        SimTools.plot_2d_gain(nf2ff_data, style="cartesian")

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        expected = 10 * np.log10(3.0) + 20 * np.log10(0.5)
        assert np.max(ydata) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_reports_the_main_lobe_without_a_beamwidth(
        self, nf2ff_data, figures, style
    ):
        """The efficiency shifts the curve without reshaping it, so the
        beamwidth would repeat the directivity plot's and is left out."""
        SimTools.plot_2d_gain(nf2ff_data, style=style)

        (annotation,) = [t.get_text() for t in figures.gcf().axes[0].texts]
        direction = float(annotation.split("Main Lobe Direction =")[1].split("°")[0])
        magnitude = float(annotation.split("Main Lobe Magnitude =")[1].split("dBi")[0])
        assert direction == pytest.approx(0.0)
        assert magnitude == pytest.approx(10 * np.log10(3.0), abs=0.01)
        assert "HPBW" not in annotation

    @pytest.mark.parametrize(
        "style", ["polar", "cartesian"], ids=["polar", "cartesian"]
    )
    def test_draws_only_the_gain_curve(self, nf2ff_data, figures, style):
        SimTools.plot_2d_gain(nf2ff_data, style=style)

        assert len(figures.gcf().axes[0].lines) == 1


# ---------------------------------------------------------------------
# plot_efficiency
# ---------------------------------------------------------------------
class FakeOpenEMSNF2FF(OpenEMSNF2FF):
    """An openEMS NF2FF box that returns ``Prad`` for every requested
    frequency, so the one-call FDTD path can be checked without a solve."""

    def __init__(self, prad):
        self.prad = np.asarray(prad, dtype=float)
        self.calls = []

    def CalcNF2FF(  # noqa: N802 - matches the openEMS method name
        self,
        output_path,
        freq,
        theta,
        phi,
        radius=1,
        center=(0, 0, 0),
        outfile=None,
        read_cached=True,
        verbose=0,
    ):
        self.calls.append(
            {
                "freq": np.atleast_1d(freq),
                "outfile": outfile,
                "read_cached": read_cached,
            }
        )
        return FakeFarField(
            E_norm=None, Dmax=None, Prad=self.prad, P_rad=None, theta=None, phi=None
        )


def resonant_band():
    """A band whose S11 dips below -10 dB only between 2.44 and 2.46 GHz."""
    freqs = np.linspace(2.4e9, 2.5e9, 11)
    s11 = np.full(freqs.size, 0.9, dtype=complex)
    s11[4:7] = [0.3, 0.05, 0.3]  # 2.44, 2.45, 2.46 GHz; -10 dB is 0.316
    return make_sim_data(freqs, np.ones(freqs.size), s11=s11)


class TestPlotEfficiency:
    def test_fdtd_transforms_every_frequency_in_one_call(self, sim_data, tmp_path):
        source = FakeOpenEMSNF2FF(prad=np.full(5, 0.5))

        SimTools.plot_efficiency(source, sim_data, output_path=tmp_path)

        (call,) = source.calls
        assert call["freq"].size == 5

    def test_samples_five_frequencies_across_the_ten_db_band_by_default(
        self, nf2ff, tmp_path
    ):
        """Efficiency only matters where the antenna is matched, and every FEM
        frequency is a field solve, so the default is a handful inside the
        -10 dB band."""
        SimTools.plot_efficiency(nf2ff, resonant_band(), output_path=tmp_path)

        solved = [call["freq"] for call in nf2ff.calls]
        assert solved == pytest.approx(np.linspace(2.44e9, 2.46e9, 5))

    def test_an_unmatched_antenna_samples_the_whole_band(self, nf2ff, tmp_path):
        """With no -10 dB band there is nothing to centre on."""
        band = make_sim_data(
            np.linspace(2.4e9, 2.5e9, 11),
            np.ones(11),
            s11=np.linspace(0.9, 0.5, 11),
        )

        SimTools.plot_efficiency(nf2ff, band, output_path=tmp_path)

        solved = [call["freq"] for call in nf2ff.calls]
        assert solved[0] == pytest.approx(2.4e9)
        assert solved[-1] == pytest.approx(2.5e9)

    def test_a_band_one_sample_wide_samples_the_whole_band(self, nf2ff, tmp_path):
        """Five copies of one frequency would plot a single point."""
        band = resonant_band()
        band.s11[[4, 6]] = 0.9

        SimTools.plot_efficiency(nf2ff, band, output_path=tmp_path)

        solved = [call["freq"] for call in nf2ff.calls]
        assert solved[0] == pytest.approx(2.4e9)
        assert solved[-1] == pytest.approx(2.5e9)

    def test_fdtd_never_reads_a_cached_result(self, sim_data, tmp_path):
        """A cached file from an earlier design would otherwise come back as
        this design's efficiency."""
        source = FakeOpenEMSNF2FF(prad=np.full(5, 0.5))

        SimTools.plot_efficiency(source, sim_data, output_path=tmp_path)

        assert source.calls[0]["read_cached"] is False

    def test_fdtd_keeps_the_sphere_compute_nf2ff_data_cached(self, sim_data, tmp_path):
        source = FakeOpenEMSNF2FF(prad=np.full(5, 0.5))

        SimTools.plot_efficiency(source, sim_data, output_path=tmp_path)

        assert source.calls[0]["outfile"] != "nf2ff_3d.h5"

    def test_fdtd_radiation_efficiency_takes_the_accepted_power_at_each_freq(
        self, tmp_path
    ):
        source = FakeOpenEMSNF2FF(prad=[0.5, 0.5, 0.5])
        band = make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 2.0, 4.0])

        result = SimTools.plot_efficiency(
            source, band, freqs=band.freqs, output_path=tmp_path
        )

        np.testing.assert_allclose(result[1], [0.5, 0.25, 0.125])

    def test_fem_solves_each_frequency_on_its_own(self, nf2ff, sim_data, tmp_path):
        """The FEM adapter takes one frequency per call."""
        SimTools.plot_efficiency(nf2ff, sim_data, num_points=3, output_path=tmp_path)

        solved = [call["freq"] for call in nf2ff.calls]
        assert solved == pytest.approx([2.4e9, 2.45e9, 2.5e9])

    def test_fem_asks_for_the_powers_without_a_pattern(self, nf2ff, sim_data, tmp_path):
        """A pattern needs a quarter-wavelength of air at every frequency,
        which the low end of the band does not have."""
        SimTools.plot_efficiency(nf2ff, sim_data, num_points=3, output_path=tmp_path)

        assert all("theta" not in call for call in nf2ff.calls)

    def test_fem_solves_only_the_given_frequencies(self, nf2ff, sim_data, tmp_path):
        SimTools.plot_efficiency(
            nf2ff, sim_data, freqs=np.array([2.4e9, 2.5e9]), output_path=tmp_path
        )

        assert [call["freq"] for call in nf2ff.calls] == [2.4e9, 2.5e9]

    def test_fem_radiation_efficiency_comes_from_ploss(self, nf2ff, sim_data, tmp_path):
        """Prad = 0.75 and Ploss = 0.25 give 75% at every frequency."""
        result = SimTools.plot_efficiency(
            nf2ff, sim_data, num_points=3, output_path=tmp_path
        )

        np.testing.assert_allclose(result[1], 0.75)

    def test_total_efficiency_derates_by_the_port_mismatch(self, tmp_path):
        """|S11| = 0.5 reflects a quarter of the incident power."""
        band = make_sim_data(
            [2.4e9, 2.45e9, 2.5e9], [1.0, 1.0, 1.0], s11=[0.0, 0.5j, 0.0]
        )

        result = SimTools.plot_efficiency(
            FakeNF2FF(), band, freqs=band.freqs, output_path=tmp_path
        )

        np.testing.assert_allclose(result[2], [0.75, 0.75 * 0.75, 0.75])

    def test_draws_both_curves_in_percent(self, nf2ff, sim_data, tmp_path, figures):
        SimTools.plot_efficiency(nf2ff, sim_data, num_points=3, output_path=tmp_path)

        radiation, total = figures.gcf().axes[0].lines
        np.testing.assert_allclose(radiation.get_ydata(), 75.0)
        np.testing.assert_allclose(total.get_ydata(), 75.0)


# ---------------------------------------------------------------------
# the 3D surfaces
# ---------------------------------------------------------------------
class TestPlot3d:
    def saved(self, tmp_path, name):
        return tmp_path / "3D_plots" / name

    def test_directivity_writes_a_vtk_mesh(self, nf2ff_data, tmp_path, no_windows):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)

        assert self.saved(tmp_path, "3D_directivity.vtk").is_file()

    def test_gain_writes_a_vtk_mesh(self, nf2ff_data, tmp_path, no_windows):
        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        assert self.saved(tmp_path, "3D_Realized_Gain.vtk").is_file()

    def test_the_mesh_matches_the_angular_grid(self, nf2ff_data, tmp_path, no_windows):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        assert mesh.n_points == 91 * 181

    def test_the_directivity_scalar_is_named_for_the_colour_bar(
        self, nf2ff_data, tmp_path, no_windows
    ):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        assert "Directivity (dBi)" in mesh.array_names

    def test_the_directivity_peak_is_dmax(self, nf2ff_data, tmp_path, no_windows):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        peak = mesh["Directivity (dBi)"].max()
        assert peak == pytest.approx(10 * np.log10(4.0), abs=1e-6)

    def test_gain_is_derated_by_the_radiation_efficiency(
        self, nf2ff_data, tmp_path, no_windows
    ):
        """With Prad = 0.75 and Ploss = 0.25 the efficiency is 75%, so the gain
        peak sits 10*log10(0.75) below the directivity peak."""
        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Realized_Gain.vtk"))
        expected = 10 * np.log10(4.0) + 10 * np.log10(0.75)
        assert mesh["Realized Gain (dBi)"].max() == pytest.approx(expected, abs=1e-6)

    def test_gain_uses_the_input_power_at_freq_without_ploss(
        self, tmp_path, no_windows
    ):
        """The openEMS box reports no Ploss, so efficiency comes from the
        port's accepted power at the pattern frequency: 2 W, not the 4 W the
        band peaks at."""
        source = FakeNF2FF(prad=0.5, with_ploss=False)
        band = make_sim_data([2.4e9, 2.45e9, 2.5e9], [1.0, 2.0, 4.0])
        nf2ff_data = far_field_from(source, band, tmp_path)

        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Realized_Gain.vtk"))
        expected = 10 * np.log10(4.0) + 10 * np.log10(0.5 / 2.0)
        assert mesh["Realized Gain (dBi)"].max() == pytest.approx(expected, abs=1e-6)

    def test_gain_is_derated_by_the_port_mismatch(self, tmp_path, no_windows):
        """|S11| = 0.5 lets 75% of the incident power in, on top of the 75%
        radiation efficiency."""
        band = make_sim_data(
            [2.4e9, 2.45e9, 2.5e9], [1.0, 1.0, 1.0], s11=[0.5, 0.5, 0.5]
        )
        nf2ff_data = far_field_from(FakeNF2FF(), band, tmp_path)

        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Realized_Gain.vtk"))
        expected = 10 * np.log10(4.0 * 0.75 * 0.75)
        assert mesh["Realized Gain (dBi)"].max() == pytest.approx(expected, abs=1e-6)

    def test_each_plot_opens_one_window(self, nf2ff_data, tmp_path, no_windows):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)
        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        assert len(no_windows) == 2

    def test_the_window_titles_name_the_quantity_and_frequency(
        self, nf2ff_data, tmp_path, no_windows
    ):
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)
        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        assert "Directivity" in no_windows[0].title
        assert "Realized Gain" in no_windows[1].title
        assert all("2.45" in window.title for window in no_windows)

    @pytest.mark.parametrize(
        "plot",
        [
            lambda result, path: SimTools.plot_3d_directivity(result, path),
            lambda result, path: SimTools.plot_3d_gain(result, path),
        ],
        ids=["directivity", "gain"],
    )
    def test_the_caller_s_e_norm_is_left_unchanged(
        self, plot, nf2ff_data, tmp_path, no_windows
    ):
        """The plots normalise a copy: an un-normalised ``E_norm`` handed in
        comes back with its original peak."""
        nf2ff_data.sphere.E_norm[:] *= 2.0

        plot(nf2ff_data, tmp_path)

        assert nf2ff_data.sphere.E_norm.max() == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("plot", "file_name", "scalar_name", "offset_db"),
        [
            (
                lambda result, path: SimTools.plot_3d_directivity(result, path),
                "3D_directivity.vtk",
                "Directivity (dBi)",
                0.0,
            ),
            (
                lambda result, path: SimTools.plot_3d_gain(result, path),
                "3D_Realized_Gain.vtk",
                "Realized Gain (dBi)",
                10 * np.log10(0.75),
            ),
        ],
        ids=["directivity", "gain"],
    )
    def test_the_pattern_follows_the_squared_field_ratio(
        self,
        plot,
        file_name,
        scalar_name,
        offset_db,
        nf2ff_data,
        tmp_path,
        no_windows,
    ):
        """Power density scales with |E|^2, so D = Dmax * (E / Emax)^2. Where
        the fixture's field is half its peak (theta = 90 deg) the pattern sits
        6 dB down, not 3 dB."""
        sphere = nf2ff_data.sphere
        field_ratio = sphere.E_norm / sphere.E_norm.max()
        expected = 10 * np.log10(4.0 * field_ratio**2) + offset_db

        plot(nf2ff_data, tmp_path)

        mesh = pv.read(self.saved(tmp_path, file_name))
        np.testing.assert_allclose(
            mesh[scalar_name], expected.ravel(order="F"), atol=1e-6
        )

    def test_directivity_then_gain_agree_on_the_peak(
        self, nf2ff_data, tmp_path, no_windows
    ):
        """Calling both, as a script plotting both does, must not shift the
        second one."""
        SimTools.plot_3d_directivity(nf2ff_data, tmp_path)
        SimTools.plot_3d_gain(nf2ff_data, tmp_path)

        directivity = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        gain = pv.read(self.saved(tmp_path, "3D_Realized_Gain.vtk"))
        assert gain["Realized Gain (dBi)"].max() == pytest.approx(
            directivity["Directivity (dBi)"].max() + 10 * np.log10(0.75), abs=1e-6
        )


# ---------------------------------------------------------------------
# add_field_dump
# ---------------------------------------------------------------------
class TestAddFieldDump:
    def test_adds_a_dump_property(self, built_inset, tmp_path):
        _antenna, sim, params, _port = built_inset

        SimTools.add_field_dump(sim, params, tmp_path)

        names = [
            sim.CSX.GetProperty(i).GetName() for i in range(sim.CSX.GetQtyProperties())
        ]
        assert any("field_dump" in name for name in names)

    def test_creates_the_dump_directory(self, built_inset, tmp_path):
        _antenna, sim, params, _port = built_inset

        SimTools.add_field_dump(sim, params, tmp_path)

        assert (tmp_path / "field_dump").is_dir()

    @pytest.mark.parametrize("dump_type", list(DumpType))
    def test_every_dump_type_is_accepted(self, built_inset, tmp_path, dump_type):
        _antenna, sim, params, _port = built_inset

        SimTools.add_field_dump(sim, params, tmp_path, dump_type=dump_type)

        names = [
            sim.CSX.GetProperty(i).GetName() for i in range(sim.CSX.GetQtyProperties())
        ]
        assert any(dump_type.value[1] in name for name in names)

    def test_the_box_encloses_the_structure(self, built_inset, tmp_path):
        """A dump smaller than the antenna records a cropped field."""
        _antenna, sim, params, _port = built_inset
        before = [np.array(p.GetBoundBox()) for p in sim.CSX.GetAllPrimitives()]

        SimTools.add_field_dump(sim, params, tmp_path)

        dump = sim.CSX.GetAllPrimitives()[-1]
        box = np.array(dump.GetBoundBox())
        for limits in before:
            for dim in (0, 1):
                assert box[0][dim] <= limits[0][dim]
                assert box[1][dim] >= limits[1][dim]

    def test_the_box_spans_the_copper_stack_in_z(self, built_inset, tmp_path):
        _antenna, sim, params, _port = built_inset

        SimTools.add_field_dump(sim, params, tmp_path)

        box = np.array(sim.CSX.GetAllPrimitives()[-1].GetBoundBox())
        assert box[0][2] == pytest.approx(0.0)
        assert box[1][2] == pytest.approx(
            params.substrate_thickness_mm + params.copper_thickness_mm
        )

    def test_user_simulation_box_sets_the_dump_extent(
        self, inset_params, sim_for, tmp_path
    ):
        bounds = [[-40.0, 30.0], [-50.0, 60.0], [-10.0, 20.0]]
        params = dataclasses.replace(inset_params, simulation_box=bounds)
        simulation = sim_for(params)

        SimTools.add_field_dump(simulation, params, tmp_path)

        dump_box = np.array(simulation.CSX.GetAllPrimitives()[-1].GetBoundBox())
        assert dump_box[0][:2] == pytest.approx([-40.0, -50.0])
        assert dump_box[1][:2] == pytest.approx([30.0, 60.0])

    def test_dump_matches_the_auto_mesh_box(self, built_inset, tmp_path):
        """With no user box, the dump spans the same x/y the mesher pads to."""
        antenna, simulation, params, _port = built_inset
        antenna.create_mesh()
        grid = simulation.CSX.GetGrid()

        SimTools.add_field_dump(simulation, params, tmp_path)

        dump_box = np.array(simulation.CSX.GetAllPrimitives()[-1].GetBoundBox())
        for dimension in (0, 1):
            grid_lines = grid.GetLines(dimension)
            assert dump_box[0][dimension] == pytest.approx(grid_lines[0], abs=1e-3)
            assert dump_box[1][dimension] == pytest.approx(grid_lines[-1], abs=1e-3)

    def test_an_empty_structure_falls_back_to_the_simulation_box(
        self, inset_params, sim_for, tmp_path
    ):
        """With no geometry to measure, the dump has to size itself from the
        parameters instead of collapsing to nothing."""
        sim = sim_for(inset_params)

        SimTools.add_field_dump(sim, inset_params, tmp_path)

        box = np.array(sim.CSX.GetAllPrimitives()[-1].GetBoundBox())
        assert box[1][0] > box[0][0]
        assert box[1][1] > box[0][1]

    def test_default_output_path_is_sim_path_under_cwd(self, built_inset):
        _antenna, sim, params, _port = built_inset

        SimTools.add_field_dump(sim, params)

        assert (Path.cwd() / "Sim_Path" / "field_dump").is_dir()


# ---------------------------------------------------------------------
# structure viewing and export wrappers
# ---------------------------------------------------------------------
class TestStructureAndExports:
    def test_write_and_show_structure_writes_the_xml(
        self, built_inset, tmp_path, monkeypatch
    ):
        _antenna, sim, _params, _port = built_inset
        launched = []
        monkeypatch.setattr(
            sim_tools.subprocess, "run", lambda cmd, **kw: launched.append(cmd)
        )

        SimTools.write_and_show_structure(sim, tmp_path)

        assert (tmp_path / "structure.xml").is_file()

    def test_write_and_show_structure_launches_the_viewer(
        self, built_inset, tmp_path, monkeypatch
    ):
        _antenna, sim, _params, _port = built_inset
        launched = []
        monkeypatch.setattr(
            sim_tools.subprocess, "run", lambda cmd, **kw: launched.append(cmd)
        )

        SimTools.write_and_show_structure(sim, tmp_path)

        assert len(launched) == 1
        assert str(tmp_path / "structure.xml") in launched[0]

    def test_export_stl_writes_into_an_stl_subdirectory(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset

        SimTools.export_stl(sim, tmp_path)

        assert (tmp_path / "stl" / "structure.stl").is_file()

    @pytest.mark.needs_cadquery
    def test_export_step_writes_into_a_step_subdirectory(self, built_inset, tmp_path):
        pytest.importorskip("cadquery")
        _antenna, sim, _params, _port = built_inset

        SimTools.export_step(sim, tmp_path)

        assert (tmp_path / "step" / "structure.step").is_file()

    @pytest.mark.needs_cadquery
    def test_xml_can_be_re_exported_to_step(self, built_inset, tmp_path, monkeypatch):
        """Re-export without re-simulating, from a previously written XML."""
        pytest.importorskip("cadquery")
        _antenna, sim, _params, _port = built_inset
        monkeypatch.setattr(sim_tools.subprocess, "run", lambda cmd, **kw: None)
        SimTools.write_and_show_structure(sim, tmp_path)

        SimTools.export_csxcad_xml_to_step(tmp_path / "structure.xml", tmp_path)

        assert (tmp_path / "step").is_dir()

    def test_create_nf2ff_returns_an_openems_box_for_fdtd(self, built_inset):
        """The box is placed on the grid, so the structure has to be meshed
        first -- on an unmeshed setup openEMS rejects it outright."""
        from openEMS.nf2ff import nf2ff as openems_nf2ff

        antenna, sim, _params, _port = built_inset
        antenna.create_mesh()

        assert isinstance(SimTools.create_nf2ff(sim), openems_nf2ff)


# ---------------------------------------------------------------------
# Default output paths and the export wrappers
# ---------------------------------------------------------------------
class TestDefaultOutputPath:
    """Every writer falls back to ``cwd / "Sim_Path"`` when no path is given.

    The autouse ``_no_stray_writes`` fixture runs each test from ``tmp_path``,
    so that is where ``Sim_Path`` lands.
    """

    @staticmethod
    def sim_path():
        return Path.cwd() / "Sim_Path"

    def test_compute_nf2ff_data(self, nf2ff, sim_data):
        SimTools.compute_nf2ff_data(nf2ff, 2.45e9, sim_data)

        assert nf2ff.calls[0]["output_path"] == self.sim_path()

    def test_plot_3d_directivity(self, nf2ff_data, no_windows):
        SimTools.plot_3d_directivity(nf2ff_data)

        assert (self.sim_path() / "3D_plots" / "3D_directivity.vtk").is_file()

    def test_plot_3d_gain(self, nf2ff_data, no_windows):
        SimTools.plot_3d_gain(nf2ff_data)

        assert (self.sim_path() / "3D_plots" / "3D_Realized_Gain.vtk").is_file()

    def test_write_and_show_structure(self, built_inset, monkeypatch):
        _antenna, sim, _params, _port = built_inset
        monkeypatch.setattr(sim_tools.subprocess, "run", lambda cmd, **kw: None)

        SimTools.write_and_show_structure(sim)

        assert (self.sim_path() / "structure.xml").is_file()

    def test_export_stl(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        SimTools.export_stl(sim)

        assert (self.sim_path() / "stl" / "structure.stl").is_file()

    def test_export_gerber(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        paths = SimTools.export_gerber(sim)

        assert paths
        assert all(path.parent == self.sim_path() / "gerber" for path in paths)

    @pytest.mark.needs_cadquery
    def test_export_step(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        SimTools.export_step(sim)

        assert (self.sim_path() / "step" / "structure.step").is_file()

    @pytest.mark.needs_cadquery
    def test_export_csxcad_xml_to_step(self, built_inset, tmp_path, monkeypatch):
        _antenna, sim, _params, _port = built_inset
        monkeypatch.setattr(sim_tools.subprocess, "run", lambda cmd, **kw: None)
        SimTools.write_and_show_structure(sim, tmp_path)

        SimTools.export_csxcad_xml_to_step(tmp_path / "structure.xml")

        assert (self.sim_path() / "step").is_dir()

    def test_print_and_save_params(self, inset_params):
        SimTools.print_and_save_params(inset_params)

        assert (self.sim_path() / "params" / "params.txt").is_file()


class TestExportGerberWrapper:
    def test_writes_into_a_gerber_subdirectory(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset

        paths = SimTools.export_gerber(sim, tmp_path)

        assert paths
        assert all(path.parent == tmp_path / "gerber" for path in paths)

    def test_the_prefix_names_the_files(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset

        paths = SimTools.export_gerber(sim, tmp_path, prefix="board")

        assert all(path.name.startswith("board") for path in paths)


# ---------------------------------------------------------------------
# show_plots
# ---------------------------------------------------------------------
class TestShowPlots:
    def test_shows_open_matplotlib_figures(self, figures, monkeypatch):
        """On the Agg backend show() is a no-op, so this only checks the
        matplotlib branch is the one taken."""
        shown = []
        monkeypatch.setattr(sim_tools.plt, "show", lambda *a, **k: shown.append(True))
        figures.figure()

        SimTools.show_plots()

        assert shown == [True]

    def test_falls_back_to_the_qt_loop_when_there_are_no_figures(self, monkeypatch):
        """A FEM mesh viewer leaves a Qt app running but no matplotlib figure;
        blocking on plt.show() there would return immediately and exit."""
        monkeypatch.setattr(sim_tools.plt, "get_fignums", lambda: [])

        class FakeApp:
            def exec(self):
                return 0

        monkeypatch.setattr(
            sim_tools.QCoreApplication, "instance", staticmethod(lambda: FakeApp())
        )

        with pytest.raises(SystemExit):
            SimTools.show_plots()

    def test_without_figures_or_a_qt_app_it_just_returns(self, monkeypatch):
        shown = []
        monkeypatch.setattr(sim_tools.plt, "get_fignums", lambda: [])
        monkeypatch.setattr(sim_tools.plt, "show", lambda *a, **k: shown.append(True))
        monkeypatch.setattr(
            sim_tools.QCoreApplication, "instance", staticmethod(lambda: None)
        )

        SimTools.show_plots()

        assert shown == [True]
