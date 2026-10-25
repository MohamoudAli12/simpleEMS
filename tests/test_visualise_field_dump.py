"""Tests for the field dump viewer (``simpleEMS.visualise_field_dump``).

The loaders run on small dump files written here in the openEMS VTK layout, and
the structure reader on a hand-written ``structure.xml``, so no solver is
needed. The window itself is never created: it needs a display.
"""

import gc
import os

import numpy as np
import pytest
import pyvista as pv

from PyQt6 import QtCore, QtWidgets

from simpleEMS import visualise_field_dump as viewer_module
from simpleEMS.visualise_field_dump import (
    _choice_label,
    _field_dump_choices,
    _interpolation_matrix,
    _load_field_dump,
    _load_mesh_structure,
    _load_structure,
    _top_on_grid,
)

# Dump coordinates are in metres; the viewer converts them to mm.
X_LINES_M = np.linspace(-2e-3, 2e-3, 9)
Y_LINES_M = np.linspace(-1e-3, 1e-3, 5)
Z_LINES_M = np.array([0.0, 0.5e-3, 1.0e-3])
GRID_SHAPE = (len(Z_LINES_M), len(Y_LINES_M), len(X_LINES_M), 3)


def write_dump(path, values, array_name="E-Field"):
    """Write ``values`` of shape (nz, ny, nx, 3) as an openEMS-style ``.vtr``."""
    grid = pv.RectilinearGrid(X_LINES_M, Y_LINES_M, Z_LINES_M)
    grid.point_data[array_name] = values.reshape(-1, 3).astype(np.float32)
    grid.save(path)


def write_frequency_dump(directory, prefix, frequency, magnitude, phase):
    """Write the ``_abs`` and ``_arg`` files of one frequency-domain dump."""
    write_dump(directory / f"{prefix}_f={frequency:.0f}_abs.vtr", magnitude)
    write_dump(directory / f"{prefix}_f={frequency:.0f}_arg.vtr", phase)


def uniform_field(rng):
    """A field strong everywhere, so cropping keeps the whole grid."""
    return 1.0 + rng.random(GRID_SHAPE)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


class TestFrequencyDomainLoading:
    def test_reconstructs_the_field_at_any_phase(self, tmp_path, rng):
        magnitude = uniform_field(rng)
        phase = rng.uniform(-np.pi, np.pi, GRID_SHAPE)
        write_frequency_dump(tmp_path, "Ef", 2.45e9, magnitude, phase)

        dump = _load_field_dump(tmp_path)

        stored_magnitude = magnitude.astype(np.float32).astype(float)
        stored_phase = phase.astype(np.float32).astype(float)
        for degrees in (0, 90, 217):
            expected = stored_magnitude * np.cos(stored_phase + np.deg2rad(degrees))
            np.testing.assert_allclose(dump.field_at(degrees), expected, atol=1e-6)

    def test_converts_the_grid_to_millimetres(self, tmp_path, rng):
        write_frequency_dump(
            tmp_path, "Ef", 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
        )

        dump = _load_field_dump(tmp_path)

        np.testing.assert_allclose(dump.x_lines, X_LINES_M * 1e3, rtol=1e-6)
        np.testing.assert_allclose(dump.z_lines, Z_LINES_M * 1e3, rtol=1e-6)

    @pytest.mark.parametrize(
        ("prefix", "symbol", "unit"),
        [("Ef", "E", "V/m"), ("Hf", "H", "A/m"), ("Jf", "J", "A/m^2")],
        ids=["electric", "magnetic", "current-density"],
    )
    def test_names_the_field_after_the_dump_type(
        self, tmp_path, rng, prefix, symbol, unit
    ):
        write_frequency_dump(
            tmp_path, prefix, 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
        )

        dump = _load_field_dump(tmp_path)

        assert (dump.symbol, dump.unit) == (symbol, unit)
        assert not dump.is_time_domain

    def test_picks_the_dumped_frequency_nearest_the_request(self, tmp_path, rng):
        for frequency in (1e9, 2e9):
            write_frequency_dump(
                tmp_path, "Ef", frequency, uniform_field(rng), np.zeros(GRID_SHAPE)
            )

        dump = _load_field_dump(tmp_path, dump_freq=2.001e9)

        assert dump.frequency == 2e9

    def test_rejects_a_frequency_that_was_not_dumped(self, tmp_path, rng):
        write_frequency_dump(
            tmp_path, "Ef", 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
        )

        with pytest.raises(ValueError, match="no Ef dump at 3 GHz"):
            _load_field_dump(tmp_path, dump_freq=3e9)

    def test_asks_for_a_frequency_when_several_were_dumped(self, tmp_path, rng):
        for frequency in (1e9, 2e9):
            write_frequency_dump(
                tmp_path, "Ef", frequency, uniform_field(rng), np.zeros(GRID_SHAPE)
            )

        with pytest.raises(ValueError, match="1 GHz, 2 GHz"):
            _load_field_dump(tmp_path)

    def test_crops_to_where_the_field_is(self, tmp_path):
        x_mesh = (X_LINES_M * 1e3)[None, None, :]
        bump = np.exp(-((x_mesh / 0.3) ** 2)) * np.ones(GRID_SHAPE[:3])
        magnitude = np.repeat(bump[..., None], 3, axis=-1)
        write_frequency_dump(tmp_path, "Ef", 1e9, magnitude, np.zeros(GRID_SHAPE))

        dump = _load_field_dump(tmp_path)

        assert len(dump.x_lines) < len(X_LINES_M)
        assert dump.field_at(0).shape == (
            len(dump.z_lines),
            len(dump.y_lines),
            len(dump.x_lines),
            3,
        )


class TestTimeDomainLoading:
    def write_run(self, directory, envelope, start_time=1_000_000.0):
        """Write one frame per envelope value, ``Ht_<timestep>.vtr``, in order."""
        for index, amplitude in enumerate(envelope):
            path = directory / f"Ht_{index * 10:010d}.vtr"
            write_dump(path, np.full(GRID_SHAPE, amplitude), array_name="H-Field")
            os.utime(path, (start_time + index, start_time + index))

    def test_plays_the_frames_in_timestep_order(self, tmp_path):
        self.write_run(tmp_path, [1.0, 2.0, 3.0])

        dump = _load_field_dump(tmp_path)

        assert dump.is_time_domain
        assert dump.timesteps == (0, 10, 20)
        np.testing.assert_allclose(dump.field_at(2), 3.0)

    def test_drops_the_silent_frames_around_the_pulse(self, tmp_path):
        self.write_run(tmp_path, [0.0, 0.001, 1.0, 2.0, 1.0, 0.001])

        dump = _load_field_dump(tmp_path)

        assert dump.timesteps == (20, 30, 40)

    def test_skips_files_left_over_from_an_earlier_run(self, tmp_path):
        self.write_run(tmp_path, [1.0, 2.0])
        leftover = tmp_path / "Ht_0000000990.vtr"
        write_dump(leftover, np.full(GRID_SHAPE, 9.0), array_name="H-Field")
        os.utime(leftover, (1000.0, 1000.0))

        dump = _load_field_dump(tmp_path)

        assert dump.timesteps == (0, 10)

    def test_thins_a_long_dump_to_evenly_spaced_frames(self, tmp_path, monkeypatch):
        monkeypatch.setattr(viewer_module, "_MAX_TIME_FRAMES", 4)
        self.write_run(tmp_path, np.ones(10))

        dump = _load_field_dump(tmp_path)

        assert dump.timesteps == (0, 30, 60, 90)


class TestChoosingTheDump:
    def test_raises_when_the_directory_holds_no_dump(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no field dump files"):
            _load_field_dump(tmp_path)

    def test_asks_which_dump_when_there_are_several(self, tmp_path, rng):
        for prefix in ("Ef", "Hf"):
            write_frequency_dump(
                tmp_path, prefix, 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
            )

        with pytest.raises(ValueError, match="efield_frequency, hfield_frequency"):
            _load_field_dump(tmp_path)

    def test_reads_the_requested_dump(self, tmp_path, rng):
        for prefix in ("Ef", "Hf"):
            write_frequency_dump(
                tmp_path, prefix, 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
            )

        assert _load_field_dump(tmp_path, "Hf").symbol == "H"

    def test_rejects_a_dump_type_it_cannot_show(self, tmp_path):
        with pytest.raises(ValueError, match="cannot show"):
            _load_field_dump(tmp_path, "SAR_f")

    def test_names_the_missing_dump_and_what_was_found(self, tmp_path, rng):
        write_frequency_dump(
            tmp_path, "Ef", 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
        )

        with pytest.raises(FileNotFoundError, match="found efield_frequency"):
            _load_field_dump(tmp_path, "Jt")


class TestListingTheDumps:
    def test_lists_every_dump_and_every_frequency(self, tmp_path, rng):
        for frequency in (2e9, 1e9):
            write_frequency_dump(
                tmp_path, "Ef", frequency, uniform_field(rng), np.zeros(GRID_SHAPE)
            )
        write_dump(tmp_path / "Ht_0000000000.vtr", uniform_field(rng), "H-Field")

        choices = _field_dump_choices(tmp_path)

        assert choices == [("Ef", 1e9), ("Ef", 2e9), ("Ht", None)]

    def test_leaves_out_dumps_it_cannot_show(self, tmp_path, rng):
        write_frequency_dump(
            tmp_path, "SAR", 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
        )

        assert _field_dump_choices(tmp_path) == []

    def test_every_listed_dump_loads(self, tmp_path, rng):
        for prefix in ("Ef", "Hf"):
            write_frequency_dump(
                tmp_path, prefix, 1e9, uniform_field(rng), np.zeros(GRID_SHAPE)
            )

        for prefix, frequency in _field_dump_choices(tmp_path):
            dump = _load_field_dump(tmp_path, prefix, frequency)
            assert (dump.prefix, dump.frequency) == (prefix, frequency)

    @pytest.mark.parametrize(
        ("prefix", "frequency", "label"),
        [
            ("Ef", 2.45e9, "Electric field - 2.45 GHz"),
            ("Hf", 1e9, "Magnetic field - 1 GHz"),
            ("Jt", None, "Total current density (curl H) - time domain"),
        ],
        ids=["electric", "magnetic", "time-domain"],
    )
    def test_names_each_dump_for_the_picker(self, prefix, frequency, label):
        assert _choice_label(prefix, frequency) == label


STRUCTURE_XML = """<?xml version="1.0"?>
<openEMS>
  <ContinuousStructure CoordSystem="0">
    <Properties>
      <Material ID="0" Name="substrate">
        <Primitives>
          <Box Priority="0"><Transformation />
            <P1 X="-10" Y="-5" Z="0" /><P2 X="10" Y="5" Z="1.6" />
          </Box>
        </Primitives>
      </Material>
      <Metal ID="1" Name="ground">
        <Primitives>
          <Box Priority="2"><Transformation />
            <P1 X="-10" Y="-5" Z="0" /><P2 X="10" Y="5" Z="-0.035" />
          </Box>
        </Primitives>
      </Metal>
      <Metal ID="2" Name="patch">
        <Primitives>
          <LinPoly Priority="0" Elevation="1.6" NormDir="2" Length="0.035">
            <Transformation />
            <Vertex X1="0" X2="0" /><Vertex X1="2" X2="0" />
            <Vertex X1="2" X2="1" /><Vertex X1="1" X2="1" />
            <Vertex X1="1" X2="2" /><Vertex X1="0" X2="2" />
          </LinPoly>
          <Cylinder Priority="6" Radius="0.3"><Transformation />
            <P1 X="5" Y="0" Z="-0.035" /><P2 X="5" Y="0" Z="1.635" />
          </Cylinder>
          <Sphere Priority="0" Radius="1"><Transformation />
            <Center X="0" Y="0" Z="0" />
          </Sphere>
        </Primitives>
      </Metal>
      <LumpedElement ID="3" Name="port_resist_1">
        <Primitives>
          <Box Priority="6"><Transformation />
            <P1 X="0" Y="0" Z="0" /><P2 X="0" Y="1" Z="1.6" />
          </Box>
        </Primitives>
      </LumpedElement>
    </Properties>
    <RectilinearGrid DeltaUnit="{delta_unit}" CoordSystem="0" />
  </ContinuousStructure>
</openEMS>
"""


class TestStructureLoading:
    @pytest.fixture
    def structure(self, tmp_path):
        path = tmp_path / "structure.xml"
        path.write_text(STRUCTURE_XML.replace("{delta_unit}", "0.001"))
        return _load_structure(path)

    def test_groups_the_bodies_by_material(self, structure):
        counts = {group: len(bodies) for group, bodies in structure.items()}

        assert counts == {"copper": 2, "ground": 1, "dielectric": 1}

    def test_extrudes_a_linpoly_from_its_elevation(self, structure):
        patch = structure["copper"][0]

        np.testing.assert_allclose(
            patch.bounds, (0.0, 2.0, 0.0, 2.0, 1.6, 1.635), atol=1e-9
        )

    def test_keeps_the_concave_corner_of_a_linpoly(self, structure):
        patch = structure["copper"][0]
        top = _top_on_grid(patch, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))

        assert top.area == pytest.approx(3.0)

    @pytest.mark.parametrize(
        ("delta_unit", "board_width_mm"),
        [("0.001", 20.0), ("1e-06", 0.02)],
        ids=["millimetres", "micrometres"],
    )
    def test_scales_drawing_units_to_millimetres(
        self, tmp_path, delta_unit, board_width_mm
    ):
        path = tmp_path / "structure.xml"
        path.write_text(STRUCTURE_XML.replace("{delta_unit}", delta_unit))

        board = _load_structure(path)["dielectric"][0]

        assert board.bounds.x_max - board.bounds.x_min == pytest.approx(board_width_mm)

    def test_warns_about_primitives_it_cannot_draw(self, tmp_path, capsys):
        path = tmp_path / "structure.xml"
        path.write_text(STRUCTURE_XML.replace("{delta_unit}", "0.001"))

        _load_structure(path)

        assert "Sphere" in capsys.readouterr().out


def write_tagged_mesh(path):
    """
    Write a FEM-style ``structure.vtk``: a substrate box (region 100) in air
    (200), a ground sheet under it and a patch sheet on top (both PEC, 300),
    and a port sheet (401), with coordinates in metres.
    """
    board = pv.Box(bounds=(0, 0.02, 0, 0.01, 0, 0.0016)).triangulate()
    substrate = board.delaunay_3d()
    air = pv.Box(bounds=(-0.01, 0.03, -0.01, 0.02, -0.01, 0.01)).delaunay_3d()
    ground = pv.Plane(center=(0.01, 0.005, 0), i_size=0.02, j_size=0.01)
    patch = pv.Plane(center=(0.01, 0.005, 0.0016), i_size=0.008, j_size=0.004)
    port = pv.Plane(
        center=(0, 0.005, 0.0008), direction=(1, 0, 0), i_size=0.0016, j_size=0.001
    )
    parts = []
    for part, region_id in (
        (substrate, 100),
        (air, 200),
        (ground.triangulate(), 300),
        (patch.triangulate(), 300),
        (port.triangulate(), 401),
    ):
        part = part.cast_to_unstructured_grid()
        part.cell_data["CellEntityIds"] = np.full(part.n_cells, region_id)
        parts.append(part)
    pv.merge(parts, merge_points=False).save(path, binary=False)


class TestMeshStructureLoading:
    @pytest.fixture
    def structure(self, tmp_path):
        path = tmp_path / "structure.vtk"
        write_tagged_mesh(path)
        return _load_mesh_structure(path)

    def test_groups_the_regions_by_material(self, structure):
        counts = {group: len(bodies) for group, bodies in structure.items()}

        assert counts == {"copper": 1, "ground": 1, "dielectric": 1}

    def test_scales_metres_to_millimetres(self, structure):
        board = structure["dielectric"][0]

        np.testing.assert_allclose(
            board.bounds, (0.0, 20.0, 0.0, 10.0, 0.0, 1.6), atol=1e-6
        )

    def test_keeps_the_patch_on_top_of_the_substrate(self, structure):
        patch = structure["copper"][0]

        np.testing.assert_allclose(
            patch.bounds, (6.0, 14.0, 3.0, 7.0, 1.6, 1.6), atol=1e-6
        )


class TestInterpolationMatrix:
    lines = (
        np.array([0.0, 1.0, 3.0]),
        np.array([-1.0, 0.5, 2.0]),
        np.array([0.0, 0.2]),
    )

    def grid_values(self, function):
        z_mesh, y_mesh, x_mesh = np.meshgrid(
            self.lines[2], self.lines[1], self.lines[0], indexing="ij"
        )
        return function(x_mesh, y_mesh, z_mesh)

    def test_reproduces_a_linear_field_inside_the_grid(self, rng):
        def linear(x, y, z):
            return 2.0 * x - 3.0 * y + 5.0 * z + 1.0

        points = rng.uniform([0.0, -1.0, 0.0], [3.0, 2.0, 0.2], size=(50, 3))

        weights = _interpolation_matrix(*self.lines, points)

        interpolated = weights @ self.grid_values(linear).ravel()
        np.testing.assert_allclose(interpolated, linear(*points.T), atol=1e-12)

    def test_gives_zero_outside_the_grid(self):
        points = np.array([[4.0, 0.0, 0.1], [1.0, 0.0, -0.5]])

        weights = _interpolation_matrix(*self.lines, points)

        np.testing.assert_array_equal(weights @ np.ones(weights.shape[1]), 0.0)


class TestViewerLifetime:
    """The window outlives the caller's reference, and Python must not free it."""

    @pytest.fixture
    def stand_in_viewer(self, tmp_path, rng, monkeypatch):
        # A bare window in place of FieldViewer: only its lifetime is under test,
        # and the offscreen platform needs no display.
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        dump_directory = tmp_path / "field_dump"
        dump_directory.mkdir()
        field = uniform_field(rng)
        write_frequency_dump(dump_directory, "Ef", 2.45e9, field, field)

        class StandInViewer(QtWidgets.QMainWindow):
            def __init__(self, *args, **kwargs):
                super().__init__()

        monkeypatch.setattr(viewer_module, "FieldViewer", StandInViewer)
        return StandInViewer

    def open_and_drop(self, output_path):
        viewer_module.view_field_dump(output_path=output_path, show=False)
        gc.collect()

    def test_keeps_the_viewer_after_the_caller_drops_it(
        self, tmp_path, stand_in_viewer
    ):
        self.open_and_drop(tmp_path)

        windows = [
            window
            for window in QtWidgets.QApplication.topLevelWidgets()
            if isinstance(window, stand_in_viewer)
        ]
        assert len(windows) == 1
        windows[0].close()

    def test_releases_the_viewer_once_it_closes(self, tmp_path, stand_in_viewer):
        self.open_and_drop(tmp_path)
        (viewer,) = (
            viewer
            for viewer in viewer_module._OPEN_VIEWERS
            if isinstance(viewer, stand_in_viewer)
        )

        viewer.close()
        QtCore.QCoreApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete
        )

        assert viewer not in viewer_module._OPEN_VIEWERS
