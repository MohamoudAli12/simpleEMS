"""Tests for how ``SimTools`` hands work to the FDTD or the FEM backend.

``run_simulation``, ``write_and_show_structure``, ``create_nf2ff`` and
``compute_sim_data`` each branch on ``sim.backend_engine``. The solvers, the
mesher and the viewer window are stubbed out here, so these tests check the
dispatch -- what is called, with what -- and never run a solve.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS import fem_backend, fem_field_dump, sim_tools  # noqa: E402
from simpleEMS.sim_tools import SimTools, setup_simulation  # noqa: E402


@pytest.fixture
def fem_sim(fr4):
    from simpleEMS.microstrip_line import MicrostripLine, MicrostripLineParams

    params = MicrostripLineParams(
        min_freq=2e9, max_freq=3e9, target_freq=2.45e9, backend_engine="FEM", **fr4
    )
    sim = setup_simulation(params)
    MicrostripLine(params, sim).build_microstrip_line()
    return sim, params


class RecordingFDTD:
    """Stands in for the openEMS object, whose methods cannot be patched."""

    def __init__(self, wander_to=None):
        self.runs = []
        self.wander_to = wander_to

    def Run(self, output_path):  # noqa: N802 - matches openEMS
        self.runs.append(output_path)
        if self.wander_to is not None:
            os.chdir(self.wander_to)


# ---------------------------------------------------------------------
# run_simulation
# ---------------------------------------------------------------------
class TestRunSimulationFDTD:
    def test_runs_the_engine_in_the_output_path(self, built_mline, tmp_path):
        _line, sim, _params, _ports = built_mline
        engine = RecordingFDTD()

        SimTools.run_simulation(sim._replace(FDTD=engine), tmp_path / "run")

        assert engine.runs == [tmp_path / "run"]

    def test_resolves_a_relative_path(self, built_mline, tmp_path):
        """openEMS compares the path with os.getcwd() after chdir-ing into it."""
        _line, sim, _params, _ports = built_mline
        engine = RecordingFDTD()

        SimTools.run_simulation(sim._replace(FDTD=engine), Path("relative"))

        assert engine.runs == [tmp_path / "relative"]
        assert engine.runs[0].is_absolute()

    def test_defaults_to_sim_path_under_cwd(self, built_mline, tmp_path):
        _line, sim, _params, _ports = built_mline
        engine = RecordingFDTD()

        SimTools.run_simulation(sim._replace(FDTD=engine))

        assert engine.runs == [tmp_path / "Sim_Path"]

    def test_restores_the_working_directory(self, built_mline, tmp_path):
        _line, sim, _params, _ports = built_mline
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        engine = RecordingFDTD(wander_to=elsewhere)

        SimTools.run_simulation(sim._replace(FDTD=engine), tmp_path / "run")

        assert Path.cwd() == tmp_path


class TestRunSimulationFEM:
    @pytest.fixture
    def recorded(self, monkeypatch):
        calls = {"sweep": [], "dumps": []}
        monkeypatch.setattr(
            fem_backend,
            "run_sweep",
            lambda csx, freqs, output_path, FEM_options: calls["sweep"].append(
                (csx, freqs, output_path, FEM_options)
            ),
        )
        monkeypatch.setattr(
            fem_field_dump,
            "write_field_dumps",
            lambda dumps, output_path: calls["dumps"].append((dumps, output_path)),
        )
        return calls

    def test_runs_the_adaptive_sweep(self, fem_sim, recorded, tmp_path):
        sim, _params = fem_sim

        SimTools.run_simulation(sim, tmp_path)

        ((csx, freqs, output_path, options),) = recorded["sweep"]
        assert csx is sim.CSX
        assert freqs is sim.freqs
        assert output_path == tmp_path
        assert options is sim.FEM_options

    def test_writes_no_dumps_unless_queued(self, fem_sim, recorded, tmp_path):
        sim, _params = fem_sim

        SimTools.run_simulation(sim, tmp_path)

        assert recorded["dumps"] == []

    def test_writes_the_queued_dumps_after_the_sweep(self, fem_sim, recorded, tmp_path):
        sim, params = fem_sim
        SimTools.add_field_dump(sim, params)

        SimTools.run_simulation(sim, tmp_path)

        ((dumps, output_path),) = recorded["dumps"]
        assert dumps == sim.FEM_field_dumps
        assert output_path == tmp_path


# ---------------------------------------------------------------------
# write_and_show_structure under FEM: the mesh viewer
# ---------------------------------------------------------------------
class TestFEMMeshViewer:
    @pytest.fixture
    def viewer(self, monkeypatch, tmp_path):
        """Stub the mesher with a two-cell grid and record the viewer calls."""
        import pyvista as pv

        msh_path = tmp_path / "mesh" / "model.msh"
        msh_path.parent.mkdir()
        grid = pv.ImageData(dimensions=(3, 2, 2)).cast_to_unstructured_grid()
        grid.cell_data["CellEntityIds"] = np.array([1, 2])
        grid.save(msh_path.with_suffix(".vtk"))

        recorded = {"build": [], "plotters": [], "themes": []}

        def fake_build_mesh(csx, freqs, output_path, FEM_options):
            recorded["build"].append((csx, output_path, FEM_options))
            return str(msh_path)

        class RecordingPlotter:
            def __init__(self):
                self.meshes, self.shown, self.calls = [], False, []
                recorded["plotters"].append(self)

            def add_mesh(self, mesh, **kwargs):
                self.meshes.append((mesh, kwargs))

            def add_axes(self):
                self.calls.append("add_axes")

            def view_xy(self):
                self.calls.append("view_xy")

            def show(self):
                self.shown = True

        monkeypatch.setattr(fem_backend, "build_mesh", fake_build_mesh)
        monkeypatch.setattr(sim_tools.pv, "Plotter", RecordingPlotter)
        monkeypatch.setattr(
            sim_tools.pv,
            "set_plot_theme",
            lambda theme: recorded["themes"].append(theme),
        )
        return recorded

    def test_meshes_and_shows_the_mesh(self, fem_sim, viewer, tmp_path):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim, tmp_path)

        ((csx, output_path, options),) = viewer["build"]
        assert csx is sim.CSX
        assert output_path == tmp_path
        assert options is sim.FEM_options
        (plotter,) = viewer["plotters"]
        assert plotter.shown
        assert plotter.calls == ["add_axes", "view_xy"]

    def test_colours_the_cells_by_region(self, fem_sim, viewer, tmp_path):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim, tmp_path)

        ((mesh, kwargs),) = viewer["plotters"][0].meshes
        assert mesh.n_cells == 2
        assert kwargs["scalars"] == "CellEntityIds"

    @pytest.mark.parametrize(
        ("mesh_style", "show_edges"),
        [("wireframe", False), ("surface", True)],
        ids=["wireframe", "surface"],
    )
    def test_edges_show_unless_drawing_a_wireframe(
        self, fem_sim, viewer, tmp_path, mesh_style, show_edges
    ):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim, tmp_path, mesh_style=mesh_style)

        ((_mesh, kwargs),) = viewer["plotters"][0].meshes
        assert kwargs["style"] == mesh_style
        assert kwargs["show_edges"] is show_edges

    def test_keeps_the_active_theme_by_default(self, fem_sim, viewer, tmp_path):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim, tmp_path)

        assert viewer["themes"] == []

    def test_applies_a_named_theme(self, fem_sim, viewer, tmp_path):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim, tmp_path, theme="document")

        assert viewer["themes"] == ["document"]

    def test_defaults_to_sim_path_under_cwd(self, fem_sim, viewer, tmp_path):
        sim, _params = fem_sim

        SimTools.write_and_show_structure(sim)

        ((_csx, output_path, _options),) = viewer["build"]
        assert output_path == tmp_path / "Sim_Path"
        assert output_path.is_dir()


# ---------------------------------------------------------------------
# create_nf2ff and compute_sim_data under FEM
# ---------------------------------------------------------------------
class TestFEMPostProcessing:
    def test_create_nf2ff_without_a_setup_is_the_fem_adapter(self):
        from simpleEMS.fem_radiation import FEMNF2FF

        assert isinstance(SimTools.create_nf2ff(), FEMNF2FF)

    def test_create_nf2ff_under_fem_is_the_fem_adapter(self, fem_sim):
        from simpleEMS.fem_radiation import FEMNF2FF

        sim, _params = fem_sim

        assert isinstance(SimTools.create_nf2ff(sim), FEMNF2FF)

    def test_compute_sim_data_reads_the_fem_results(
        self, fem_sim, monkeypatch, tmp_path
    ):
        sim, _params = fem_sim
        received = []
        monkeypatch.setattr(
            fem_backend,
            "compute_sim_data",
            lambda freqs, charac_imp, output_path: (
                received.append((freqs, charac_imp, output_path)) or "sim-data"
            ),
        )

        result = SimTools.compute_sim_data(sim, port=None, output_path=tmp_path)

        assert result == "sim-data"
        assert received == [(sim.freqs, sim.charac_imp, tmp_path)]

    def test_compute_sim_data_defaults_to_sim_path(
        self, fem_sim, monkeypatch, tmp_path
    ):
        sim, _params = fem_sim
        received = []
        monkeypatch.setattr(
            fem_backend,
            "compute_sim_data",
            lambda freqs, charac_imp, output_path: received.append(output_path),
        )

        SimTools.compute_sim_data(sim, port=None)

        assert received == [tmp_path / "Sim_Path"]
