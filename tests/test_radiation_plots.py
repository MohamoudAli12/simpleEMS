"""Tests for the radiation, 3D-visualisation and orchestration parts of SimTools.

``test_plotting.py`` covers the 2D S-parameter family. This file covers the
rest: the polar radiation cuts, the PyVista surfaces, the field dump, the
export wrappers, and ``run_all_post_processing``, which strings them all
together and is what most example scripts actually call.

Everything here is driven from a synthetic far field rather than a solve. The
plots take an ``nf2ff``-shaped object and never look at where it came from, so
a stub with a known analytic pattern exercises the same code an antenna would
and lets the tests assert on numbers that can be worked out by hand.

Two things are always stubbed: ``BackgroundPlotter``, which opens a Qt window
per 3D plot, and ``show_plots``, which blocks until those windows are closed.
"""

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

from simpleEMS import sim_tools  # noqa: E402
from simpleEMS.sim_tools import DumpType, SimTools  # noqa: E402


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


@pytest.fixture
def nf2ff():
    return FakeNF2FF()


@pytest.fixture
def far_field_3d(nf2ff, tmp_path):
    """What ``compute_nf2ff_3d`` hands to the 3D plots."""
    return SimTools.compute_nf2ff_3d(nf2ff, 2.45e9, tmp_path)


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
# plot_2d_rad_pattern
# ---------------------------------------------------------------------
class TestPlot2dRadPattern:
    def test_opens_one_figure_with_two_polar_axes(self, nf2ff, tmp_path, figures):
        """The xz and xy cuts are drawn side by side in one figure."""
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        assert len(figures.get_fignums()) == 1
        assert len(figures.gcf().axes) == 2

    def test_both_axes_are_polar(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        assert all(ax.name == "polar" for ax in figures.gcf().axes)

    def test_asks_for_both_principal_planes(self, nf2ff, tmp_path):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        first, second = nf2ff.calls
        # xz cut: theta swept at phi = 0
        assert first["phi"].tolist() == [0]
        assert first["theta"].min() == -180.0
        # xy cut: phi swept at theta = 90
        assert second["theta"].tolist() == [90]
        assert second["phi"].min() == -180.0

    def test_sweeps_a_full_circle(self, nf2ff, tmp_path):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        theta = nf2ff.calls[0]["theta"]
        assert theta.min() == -180.0
        assert theta.max() == 180.0

    def test_the_pattern_is_normalised_to_its_peak(self, nf2ff, tmp_path, figures):
        """A polar cut is always plotted relative to its own maximum, so the
        peak sits at 0 dB whatever the absolute field level was."""
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(0.0)

    def test_the_broadside_peak_is_at_theta_zero(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        line = figures.gcf().axes[0].lines[0]
        peak_theta = np.rad2deg(line.get_xdata()[np.argmax(line.get_ydata())])
        assert peak_theta == pytest.approx(0.0)

    def test_each_cut_is_labelled(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        labels = [ax.lines[0].get_label() for ax in figures.gcf().axes]
        assert labels == ["xz-plane", "xy-plane"]

    def test_the_frequency_is_in_the_title(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        assert "2.45" in figures.gcf()._suptitle.get_text()

    def test_zero_is_north_and_angles_run_clockwise(self, nf2ff, tmp_path, figures):
        """Antenna convention: theta = 0 points up the page."""
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        for ax in figures.gcf().axes:
            assert ax.get_theta_offset() == pytest.approx(np.pi / 2)
            assert ax.get_theta_direction() == -1

    def test_an_array_of_frequencies_is_rejected(self, nf2ff, tmp_path):
        """One cut is one frequency; an array would silently plot the first."""
        with pytest.raises(TypeError, match="only one frequency"):
            SimTools.plot_2d_rad_pattern(nf2ff, np.array([2.4e9, 2.5e9]), tmp_path)

    def test_read_cached_is_passed_through(self, nf2ff, tmp_path):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path, read_cached=True)

        assert all(call["read_cached"] for call in nf2ff.calls)

    def test_the_output_path_is_passed_through(self, nf2ff, tmp_path):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9, tmp_path)

        assert nf2ff.calls[0]["output_path"] == tmp_path

    def test_default_output_path_is_sim_path_under_cwd(self, nf2ff):
        SimTools.plot_2d_rad_pattern(nf2ff, 2.45e9)

        assert nf2ff.calls[0]["output_path"] == Path.cwd() / "Sim_Path"


# ---------------------------------------------------------------------
# plot_2d_directivity
# ---------------------------------------------------------------------
class TestPlot2dDirectivity:
    def test_opens_one_polar_figure(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)

        assert len(figures.get_fignums()) == 1
        assert figures.gcf().axes[0].name == "polar"

    def test_the_peak_is_the_max_directivity(self, nf2ff, tmp_path, figures):
        """The curve is the normalised pattern lifted by Dmax, so its peak is
        Dmax in dBi -- 10*log10(4) = 6.02 dBi here."""
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)

        ydata = figures.gcf().axes[0].lines[0].get_ydata()
        assert np.max(ydata) == pytest.approx(10 * np.log10(4.0), abs=1e-6)

    def test_the_beamwidth_is_annotated(self, nf2ff, tmp_path, figures):
        texts = []
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)
        for text in figures.gcf().axes[0].texts:
            texts.append(text.get_text())

        assert any("HPBW" in t for t in texts)

    def test_the_computed_beamwidth_matches_the_pattern(self, nf2ff, tmp_path, figures):
        """cos^2(theta/2) falls to -3 dB at theta = +-65.4 deg, so the
        half-power beamwidth is about 131 deg."""
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)

        annotation = next(
            t.get_text() for t in figures.gcf().axes[0].texts if "HPBW" in t.get_text()
        )
        value = float(annotation.split("=")[1].split("°")[0])
        assert value == pytest.approx(130.8, abs=1.0)

    def test_the_main_lobe_direction_is_reported(self, nf2ff, tmp_path, figures):
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)

        annotation = next(
            t.get_text() for t in figures.gcf().axes[0].texts if "HPBW" in t.get_text()
        )
        direction = float(annotation.split("Main Lobe Direction =")[1].split("°")[0])
        assert direction == pytest.approx(0.0)

    def test_a_pattern_with_no_half_power_point_is_rejected(self, tmp_path):
        """An isotropic pattern never drops 3 dB, so no beamwidth exists and
        the annotation would otherwise be nonsense."""

        class Isotropic(FakeNF2FF):
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

        with pytest.raises(ValueError, match="HPBW could not be determined"):
            SimTools.plot_2d_directivity(Isotropic(), 2.45e9, tmp_path)

    def test_an_array_of_frequencies_is_rejected(self, nf2ff, tmp_path):
        with pytest.raises(TypeError, match="only one frequency"):
            SimTools.plot_2d_directivity(nf2ff, np.array([2.4e9, 2.5e9]), tmp_path)

    def test_uses_a_finer_sweep_than_the_pattern_plot(self, nf2ff, tmp_path):
        """The beamwidth is read off the sampled curve, so the grid has to be
        fine enough to place the -3 dB crossing accurately."""
        SimTools.plot_2d_directivity(nf2ff, 2.45e9, tmp_path)

        theta = nf2ff.calls[0]["theta"]
        assert np.diff(theta).max() == pytest.approx(0.1)


# ---------------------------------------------------------------------
# compute_nf2ff_3d
# ---------------------------------------------------------------------
class TestComputeNF2FF3d:
    def test_sweeps_the_full_sphere(self, nf2ff, tmp_path):
        SimTools.compute_nf2ff_3d(nf2ff, 2.45e9, tmp_path)

        call = nf2ff.calls[0]
        assert call["theta"][0] == 0 and call["theta"][-1] == 180
        assert call["phi"][0] == 0 and call["phi"][-1] == 360

    def test_grid_is_two_degree_steps(self, nf2ff, tmp_path):
        SimTools.compute_nf2ff_3d(nf2ff, 2.45e9, tmp_path)

        call = nf2ff.calls[0]
        assert call["theta"].size == 91
        assert call["phi"].size == 181

    def test_returns_the_far_field_result(self, nf2ff, tmp_path):
        result = SimTools.compute_nf2ff_3d(nf2ff, 2.45e9, tmp_path)

        assert result.E_norm.shape == (91, 181)

    def test_an_array_of_frequencies_is_rejected(self, nf2ff, tmp_path):
        with pytest.raises(TypeError, match="only one frequency"):
            SimTools.compute_nf2ff_3d(nf2ff, np.array([2.4e9, 2.5e9]), tmp_path)

    def test_default_output_path_is_sim_path_under_cwd(self, nf2ff):
        SimTools.compute_nf2ff_3d(nf2ff, 2.45e9)

        assert nf2ff.calls[0]["output_path"] == Path.cwd() / "Sim_Path"


# ---------------------------------------------------------------------
# the 3D surfaces
# ---------------------------------------------------------------------
class TestPlot3d:
    def saved(self, tmp_path, name):
        return tmp_path / "3D_plots" / name

    def test_directivity_writes_a_vtk_mesh(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        assert self.saved(tmp_path, "3D_directivity.vtk").is_file()

    def test_gain_writes_a_vtk_mesh(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_gain(far_field_3d, 2.45e9, 1.0, tmp_path)

        assert self.saved(tmp_path, "3D_Gain.vtk").is_file()

    def test_power_writes_a_vtk_mesh(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_power(far_field_3d, 2.45e9, tmp_path)

        assert self.saved(tmp_path, "3D_Power.vtk").is_file()

    def test_the_mesh_matches_the_angular_grid(
        self, far_field_3d, tmp_path, no_windows
    ):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        assert mesh.n_points == 91 * 181

    def test_the_directivity_scalar_is_named_for_the_colour_bar(
        self, far_field_3d, tmp_path, no_windows
    ):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        assert "Directivity (dBi)" in mesh.array_names

    def test_the_directivity_peak_is_dmax(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        peak = mesh["Directivity (dBi)"].max()
        assert peak == pytest.approx(10 * np.log10(4.0), abs=1e-6)

    def test_gain_is_derated_by_the_radiation_efficiency(
        self, far_field_3d, tmp_path, no_windows
    ):
        """With Prad = 0.75 and Ploss = 0.25 the efficiency is 75%, so the gain
        peak sits 10*log10(0.75) below the directivity peak."""
        SimTools.plot_3d_gain(far_field_3d, 2.45e9, 1.0, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Gain.vtk"))
        expected = 10 * np.log10(4.0) + 10 * np.log10(0.75)
        assert mesh["Gain (dBi)"].max() == pytest.approx(expected, abs=1e-6)

    def test_gain_falls_back_to_input_power_without_ploss(self, tmp_path, no_windows):
        """The openEMS box reports no Ploss, so efficiency has to come from the
        port's input power instead."""
        source = FakeNF2FF(prad=0.5, with_ploss=False)
        result = SimTools.compute_nf2ff_3d(source, 2.45e9, tmp_path)

        SimTools.plot_3d_gain(result, 2.45e9, 2.0, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Gain.vtk"))
        expected = 10 * np.log10(4.0) + 10 * np.log10(0.5 / 2.0)
        assert mesh["Gain (dBi)"].max() == pytest.approx(expected, abs=1e-6)

    def test_power_is_normalised_to_its_peak(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_power(far_field_3d, 2.45e9, tmp_path)

        mesh = pv.read(self.saved(tmp_path, "3D_Power.vtk"))
        assert mesh["Power (dB)"].max() == pytest.approx(0.0, abs=1e-9)

    def test_each_plot_opens_one_window(self, far_field_3d, tmp_path, no_windows):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)
        SimTools.plot_3d_gain(far_field_3d, 2.45e9, 1.0, tmp_path)
        SimTools.plot_3d_power(far_field_3d, 2.45e9, tmp_path)

        assert len(no_windows) == 3

    def test_the_window_titles_name_the_quantity_and_frequency(
        self, far_field_3d, tmp_path, no_windows
    ):
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        assert "Directivity" in no_windows[0].title
        assert "2.45" in no_windows[0].title

    def test_an_array_of_frequencies_is_rejected(
        self, far_field_3d, tmp_path, no_windows
    ):
        with pytest.raises(TypeError, match="only one frequency"):
            SimTools.plot_3d_directivity(
                far_field_3d, np.array([2.4e9, 2.5e9]), tmp_path
            )

    def test_the_result_is_normalised_in_place(
        self, far_field_3d, tmp_path, no_windows
    ):
        """Documented here because it is surprising: the 3D plots divide the
        caller's own ``E_norm`` by its peak rather than a copy. It is
        idempotent, so plotting directivity then gain still agrees -- but the
        array handed in does not survive the call unchanged."""
        before = far_field_3d.E_norm.max()

        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)

        assert before == pytest.approx(1.0)  # already normalised in this fixture
        assert far_field_3d.E_norm.max() == pytest.approx(1.0)

    def test_directivity_then_gain_agree_on_the_peak(
        self, far_field_3d, tmp_path, no_windows
    ):
        """The consequence of the in-place normalisation above: calling both,
        as run_all_post_processing does, must not shift the second one."""
        SimTools.plot_3d_directivity(far_field_3d, 2.45e9, tmp_path)
        SimTools.plot_3d_gain(far_field_3d, 2.45e9, 1.0, tmp_path)

        directivity = pv.read(self.saved(tmp_path, "3D_directivity.vtk"))
        gain = pv.read(self.saved(tmp_path, "3D_Gain.vtk"))
        assert gain["Gain (dBi)"].max() == pytest.approx(
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


# ---------------------------------------------------------------------
# run_all_post_processing
# ---------------------------------------------------------------------
class TestRunAllPostProcessing:
    @pytest.fixture
    def ran(self, built_inset, nf2ff, tmp_path, no_windows, monkeypatch):
        """Drive the whole pipeline once, with the blocking bits stubbed."""
        _antenna, sim, params, _port = built_inset
        monkeypatch.setattr(SimTools, "show_plots", staticmethod(lambda: None))

        freqs = np.linspace(2.2e9, 2.7e9, 21)
        s11 = 0.2 * np.exp(1j * np.linspace(0, np.pi, 21))
        vswr = (1 + np.abs(s11)) / (1 - np.abs(s11))
        z11 = 50 * (1 + s11) / (1 - s11)
        far_field_3d = SimTools.compute_nf2ff_3d(nf2ff, params.main_freq, tmp_path)

        SimTools.run_all_post_processing(
            sim,
            freqs,
            s11,
            vswr,
            z11,
            1.0,
            nf2ff,
            far_field_3d,
            params,
            output_path=tmp_path,
        )
        return tmp_path, sim, params

    def test_saves_the_two_dimensional_plots(self, ran):
        out, _sim, _params = ran

        saved = sorted((out / "plots").glob("plot_*.png"))
        assert len(saved) >= 6  # s-param, smith, vswr, impedance, pattern, directivity

    def test_writes_the_three_dimensional_meshes(self, ran):
        out, _sim, _params = ran

        for name in ("3D_directivity.vtk", "3D_Gain.vtk", "3D_Power.vtk"):
            assert (out / "3D_plots" / name).is_file()

    def test_exports_touchstone(self, ran):
        out, _sim, _params = ran

        assert list((out / "touchstone").glob("*.s1p"))

    def test_exports_gerber(self, ran):
        out, _sim, _params = ran

        assert list(out.rglob("*.gbr"))

    def test_exports_stl(self, ran):
        """Regression: this used to be called as ``export_stl(output_path)``,
        which bound the path to the ``sim`` parameter and raised
        ``AttributeError: 'PosixPath' object has no attribute 'CSX'``."""
        out, _sim, _params = ran

        assert (out / "stl" / "structure.stl").is_file()

    def test_the_gerber_ignores_the_ground_layer(self, ran):
        """A ground pour would swamp the fabrication output."""
        out, _sim, _params = ran

        gerber = next(out.rglob("*.gbr")).read_text()
        assert "ground" not in gerber

    def test_radiation_plots_use_the_main_frequency(self, ran, nf2ff):
        _out, _sim, params = ran

        assert all(call["freq"] == params.main_freq for call in nf2ff.calls)
