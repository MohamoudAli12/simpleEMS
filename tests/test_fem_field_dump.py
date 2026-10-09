"""Tests for the FEM field dumps (``simpleEMS.fem_field_dump``).

The reading, sampling, curl and file-writing tests run on hand-built
tetrahedra and on field files gmsh writes in the format GetDP's ``Get_Fields``
uses, so they need neither GetDP nor CSXCAD. ``write_field_dumps`` runs with
the GetDP solve stubbed out. The end-to-end test solves a coarse microstrip
line and checks the files ``SimTools.add_field_dump`` asks for appear.
"""

import json
import math

import numpy as np
import pytest

# gmsh dlopen()s X/GL libraries at import; on a bare host that raises OSError,
# which importorskip does not catch.
try:
    import gmsh
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

pv = pytest.importorskip("pyvista")

from simpleEMS import fem_field_dump  # noqa: E402
from simpleEMS.fem_field_dump import (  # noqa: E402
    FieldDumpRequest,
    _curl,
    _dump_lines,
    _exploded_mesh,
    _frequency_label,
    _read_volume_fields,
    _sample,
    _start_gmsh,
    _VolumeFields,
    _write_openems_files,
    write_field_dumps,
)

# Two tetrahedra sharing the face (1, 2, 3), which lies in the plane
# x + y + z = 1: LOWER is the corner tetrahedron at the origin, UPPER sits on
# the other side of that face.
POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)
LOWER, UPPER = [0, 1, 2, 3], [1, 2, 3, 4]


def uniform(*fields):
    """``(n_tets, 4, 3)`` complex field, constant over each tetrahedron."""
    return np.array([[field] * 4 for field in fields], dtype=complex)


def volume_fields(tets, efield, hfield=None):
    tets = np.array(tets)
    if hfield is None:
        hfield = np.zeros_like(efield)
    return _VolumeFields(points=POINTS, tets=tets, efield=efield, hfield=hfield)


def request(prefix="Ef", freq=1e9, box=(0, 0, 0, 1, 1, 1), lines=None):
    if lines is None:
        lines = ((0.0, 0.5, 1.0),) * 3
    return FieldDumpRequest(prefix=prefix, freq=freq, box=box, lines=lines)


def write_pos(path, tets, values, data_tets=None):
    """Write a field file the way GetDP's ``Get_Fields`` writes one.

    The file carries its own tetrahedral mesh, then the field at each element's
    nodes as two view steps: the real part, then the imaginary part.
    ``data_tets`` (1-based element tags) limits the field data to some of the
    elements.
    """
    tags = list(range(1, len(tets) + 1))
    if data_tets is None:
        data_tets = tags
    _start_gmsh()
    try:
        gmsh.model.add("field")
        entity = gmsh.model.addDiscreteEntity(3)
        gmsh.model.mesh.addNodes(3, entity, [1, 2, 3, 4, 5], POINTS.ravel())
        gmsh.model.mesh.addElementsByType(entity, 4, tags, (np.array(tets) + 1).ravel())
        view = gmsh.view.add("field")
        for step, part in enumerate((values.real, values.imag)):
            data = [part[tag - 1].ravel().tolist() for tag in data_tets]
            gmsh.view.addModelData(
                view, step, "field", "ElementNodeData", data_tets, data, numComponents=3
            )
        gmsh.option.setNumber("PostProcessing.SaveMesh", 1)
        msh_path = path.with_suffix(".msh")
        gmsh.view.write(view, str(msh_path))
    finally:
        gmsh.finalize()
    msh_path.rename(path)
    return path


# ---------------------------------------------------------------------
# FieldDumpRequest
# ---------------------------------------------------------------------
class TestFieldDumpRequest:
    @pytest.mark.parametrize("prefix", ["Ef", "Hf", "Jf"])
    def test_accepts_the_frequency_domain_prefixes(self, prefix):
        assert request(prefix=prefix).prefix == prefix

    @pytest.mark.parametrize("prefix", ["Et", "If", "SAR_f"])
    def test_refuses_what_the_fem_backend_cannot_dump(self, prefix):
        with pytest.raises(ValueError, match="prefix must be one of"):
            request(prefix=prefix)

    def test_equal_requests_hash_alike(self):
        """``write_field_dumps`` keys its results on the request."""
        assert hash(request()) == hash(request())


# ---------------------------------------------------------------------
# Reading GetDP's field files
# ---------------------------------------------------------------------
class TestReadVolumeFields:
    def test_start_gmsh_replaces_a_running_session(self):
        gmsh.initialize()

        _start_gmsh()

        assert gmsh.isInitialized()
        gmsh.finalize()

    def test_reads_the_mesh_and_the_complex_fields(self, tmp_path):
        efield = uniform([1 + 2j, 0, 0], [0, 3 - 1j, 0])
        hfield = uniform([0, 0, 1j], [0, 0, 2])
        e_pos = write_pos(tmp_path / "e.pos", [LOWER, UPPER], efield)
        h_pos = write_pos(tmp_path / "h.pos", [LOWER, UPPER], hfield)

        _start_gmsh()
        try:
            fields = _read_volume_fields(e_pos, h_pos)
        finally:
            gmsh.finalize()

        assert fields.points == pytest.approx(POINTS)
        assert fields.tets.tolist() == [LOWER, UPPER]
        assert fields.efield == pytest.approx(efield)
        assert fields.hfield == pytest.approx(hfield)

    def test_refuses_field_data_that_skips_an_element(self, tmp_path):
        efield = uniform([1, 0, 0], [0, 1, 0])
        e_pos = write_pos(tmp_path / "e.pos", [LOWER, UPPER], efield, data_tets=[2])

        _start_gmsh()
        try:
            with pytest.raises(RuntimeError, match="does not follow its mesh"):
                _read_volume_fields(e_pos, e_pos)
        finally:
            gmsh.finalize()

    def test_refuses_e_and_h_on_different_meshes(self, tmp_path):
        e_pos = write_pos(
            tmp_path / "e.pos", [LOWER, UPPER], uniform([1, 0, 0], [0, 1, 0])
        )
        h_pos = write_pos(tmp_path / "h.pos", [LOWER], uniform([1, 0, 0]))

        _start_gmsh()
        try:
            with pytest.raises(RuntimeError, match="different meshes"):
                _read_volume_fields(e_pos, h_pos)
        finally:
            gmsh.finalize()


# ---------------------------------------------------------------------
# Sampling on a rectilinear grid
# ---------------------------------------------------------------------
class TestSample:
    def test_exploded_mesh_gives_every_tetrahedron_its_own_points(self):
        fields = volume_fields([LOWER, UPPER], uniform([1, 0, 0], [0, 1, 0]))

        mesh = _exploded_mesh(fields)

        assert mesh.n_cells == 2
        assert mesh.n_points == 8
        assert mesh.point_data["E"].shape == (8, 6)

    def test_inside_a_tetrahedron_gets_its_value(self):
        fields = volume_fields([LOWER, UPPER], uniform([1 + 1j, 0, 0], [0, 2, 0]))
        lines = (np.array([0.1, 0.2]), np.array([0.1]), np.array([0.1]))

        field = _sample(_exploded_mesh(fields), "E", lines)

        assert field.shape == (2, 1, 1, 3)
        assert field[:, 0, 0] == pytest.approx(np.array([[1 + 1j, 0, 0]] * 2))

    def test_a_point_on_an_upper_face_still_gets_a_value(self):
        """Nothing lies past the face, so the point is sampled nudged back."""
        fields = volume_fields([LOWER], uniform([1, 0, 0]))
        lines = (np.array([0.5]), np.array([0.25]), np.array([0.25]))

        field = _sample(_exploded_mesh(fields), "E", lines)

        assert field[0, 0, 0] == pytest.approx([1, 0, 0])

    def test_a_point_off_the_mesh_is_zero(self):
        fields = volume_fields([LOWER], uniform([1, 0, 0]))
        lines = (np.array([0.1, 5.0]), np.array([0.1]), np.array([0.1]))

        field = _sample(_exploded_mesh(fields), "E", lines)

        assert field[1, 0, 0] == pytest.approx([0, 0, 0])

    def test_samples_h_by_name(self):
        fields = volume_fields([LOWER], uniform([1, 0, 0]), hfield=uniform([0, 0, 1j]))
        lines = (np.array([0.1]), np.array([0.1]), np.array([0.1]))

        field = _sample(_exploded_mesh(fields), "H", lines)

        assert field[0, 0, 0] == pytest.approx([0, 0, 1j])


# ---------------------------------------------------------------------
# Curl
# ---------------------------------------------------------------------
class TestCurl:
    @staticmethod
    def grid(axis_lines):
        x_lines, y_lines, z_lines = axis_lines
        return np.meshgrid(x_lines, y_lines, z_lines, indexing="ij")

    def test_a_rotating_field_has_a_constant_curl(self):
        """curl (-y, x, 0) = (0, 0, 2)."""
        lines = (np.linspace(0, 1, 4), np.linspace(0, 2, 5), np.linspace(0, 1, 3))
        x_values, y_values, _z_values = self.grid(lines)
        field = np.stack([-y_values, x_values, np.zeros_like(x_values)], axis=-1)

        curl = _curl(field.astype(complex), lines)

        assert curl[..., 0] == pytest.approx(0)
        assert curl[..., 1] == pytest.approx(0)
        assert curl[..., 2] == pytest.approx(2)

    def test_a_single_line_axis_contributes_no_derivative(self):
        lines = (np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.array([0.0]))
        x_values, _y_values, _z_values = self.grid(lines)
        field = np.stack(
            [np.zeros_like(x_values), np.zeros_like(x_values), x_values], axis=-1
        )

        curl = _curl(field, lines)

        assert curl.shape == field.shape
        assert curl[..., 1] == pytest.approx(-1)


# ---------------------------------------------------------------------
# Choosing the dump lines
# ---------------------------------------------------------------------
class TestDumpLines:
    LINES = ((0.0, 1.0, 2.0, 3.0, 4.0),) * 3
    DOMAIN = (0, 0, 0, 4, 4, 4)

    def test_keeps_the_lines_in_the_box(self):
        dump = request(box=(1, 1, 1, 3, 3, 3), lines=self.LINES)

        lines, box = _dump_lines(dump, self.DOMAIN, margin=0)

        for axis in range(3):
            assert lines[axis] == pytest.approx([1, 2, 3])
            assert box[axis] == slice(0, 3)

    def test_a_margin_adds_a_line_past_each_face(self):
        dump = request(box=(1, 1, 1, 3, 3, 3), lines=self.LINES)

        lines, box = _dump_lines(dump, self.DOMAIN, margin=1)

        assert lines[0] == pytest.approx([0, 1, 2, 3, 4])
        assert lines[0][box[0]] == pytest.approx([1, 2, 3])

    def test_the_margin_stops_at_the_domain(self):
        dump = request(box=(0, 0, 0, 4, 4, 4), lines=self.LINES)

        lines, box = _dump_lines(dump, self.DOMAIN, margin=1)

        assert lines[0] == pytest.approx([0, 1, 2, 3, 4])
        assert box[0] == slice(0, 5)

    def test_drops_lines_outside_the_domain(self):
        dump = request(box=(0, 0, 0, 4, 4, 4), lines=self.LINES)

        lines, _box = _dump_lines(dump, (0, 0, 0, 2, 4, 4), margin=0)

        assert lines[0] == pytest.approx([0, 1, 2])
        assert lines[1] == pytest.approx([0, 1, 2, 3, 4])

    def test_a_box_with_no_line_is_refused(self):
        dump = request(box=(10, 0, 0, 11, 4, 4), lines=self.LINES)

        with pytest.raises(ValueError, match="no x mesh line"):
            _dump_lines(dump, (0, 0, 0, 20, 4, 4), margin=0)


# ---------------------------------------------------------------------
# Writing the openEMS-style files
# ---------------------------------------------------------------------
class TestWriteOpenemsFiles:
    @pytest.mark.parametrize(
        ("freq", "label"),
        [(2.45e9, "2450000000"), (1.5, "1.500000")],
        ids=["whole-hertz", "fractional"],
    )
    def test_frequency_label_matches_openems(self, freq, label):
        assert _frequency_label(freq) == label

    @pytest.fixture
    def written(self, tmp_path):
        lines = (np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), np.array([0.0]))
        field = np.zeros((2, 3, 1, 3), dtype=complex)
        field[..., 0] = 1j
        field[1, 2, 0, 2] = 2.0
        stem = tmp_path / "Ef_f=1000"
        return _write_openems_files(field, lines, "E-Field", stem), field

    def test_writes_21_phases_then_magnitude_and_phase(self, written):
        paths, _field = written

        names = [path.name for path in paths]

        assert len(names) == 23
        assert names[0] == "Ef_f=1000_p=000.vtr"
        assert names[-3] == "Ef_f=1000_p=342.vtr"
        assert names[-2:] == ["Ef_f=1000_abs.vtr", "Ef_f=1000_arg.vtr"]
        assert all(path.exists() for path in paths)

    def test_phase_frames_are_the_real_part_rotated(self, written):
        paths, field = written
        angle = 2 * math.pi * 5 / 21

        frame = pv.read(paths[5]).point_data["E-Field"]

        expected = np.real(field * np.exp(1j * angle))
        # VTK order: x fastest, then y, then z
        assert frame == pytest.approx(
            expected.transpose(2, 1, 0, 3).reshape(-1, 3), abs=1e-6
        )

    def test_magnitude_and_phase_files(self, written):
        paths, field = written

        magnitude = pv.read(paths[-2]).point_data["E-Field"]
        phase = pv.read(paths[-1]).point_data["E-Field"]

        flat = field.transpose(2, 1, 0, 3).reshape(-1, 3)
        assert magnitude == pytest.approx(np.abs(flat), abs=1e-6)
        assert phase == pytest.approx(np.angle(flat), abs=1e-6)


# ---------------------------------------------------------------------
# write_field_dumps, with the GetDP solve stubbed out
# ---------------------------------------------------------------------
class TestWriteFieldDumps:
    LINES = ((0.1, 0.15, 0.2),) * 3
    BOX = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2)

    @pytest.fixture
    def solves(self, tmp_path, monkeypatch):
        """Stub GetDP and its field files; return the recorded solves."""
        (tmp_path / "fem_mesh.json").write_text(
            json.dumps(
                {
                    "port_numbers": [2, 1],
                    "pro_path": "model.pro",
                    "msh_path": "model.msh",
                    "bbox": [0, 0, 0, 1, 1, 1],
                }
            )
        )
        calls = []

        def fake_run_getdp(pro_path, msh_path, output_path, parameters, *args, **kw):
            calls.append(parameters)

        def fake_read(e_pos, h_pos):
            return volume_fields([LOWER], uniform([1, 0, 0]), hfield=uniform([0, 0, 1]))

        monkeypatch.setattr(fem_field_dump.fem_solver, "run_getdp", fake_run_getdp)
        monkeypatch.setattr(fem_field_dump, "_read_volume_fields", fake_read)
        return calls

    def dump(self, prefix, freq):
        return FieldDumpRequest(prefix, freq, self.BOX, self.LINES)

    def test_needs_a_simulation_first(self, tmp_path):
        with pytest.raises(RuntimeError, match="run the simulation first"):
            write_field_dumps([self.dump("Ef", 1e9)], tmp_path)

    def test_solves_once_per_frequency_driving_the_lowest_port(self, tmp_path, solves):
        dumps = [self.dump("Ef", 2e9), self.dump("Hf", 1e9), self.dump("Jf", 2e9)]

        write_field_dumps(dumps, tmp_path, verbose=False)

        assert solves == [
            {"FREQ": 1e9, "ACTIVE_PORT": 1},
            {"FREQ": 2e9, "ACTIVE_PORT": 1},
        ]

    def test_returns_every_file_request_by_request(self, tmp_path, solves):
        dumps = [self.dump("Hf", 2e9), self.dump("Ef", 1e9)]

        paths = write_field_dumps(dumps, tmp_path, verbose=False)

        assert len(paths) == 46
        assert paths[0].name.startswith("Hf_f=2000000000_")
        assert paths[23].name.startswith("Ef_f=1000000000_")
        assert all(path.parent == tmp_path / "field_dump" for path in paths)

    def test_samples_the_requested_field(self, tmp_path, solves):
        (efield_path, *_rest) = write_field_dumps(
            [self.dump("Ef", 1e9)], tmp_path, verbose=False
        )

        frame = pv.read(efield_path).point_data["E-Field"]

        assert frame[:, 0] == pytest.approx(1.0)
        assert frame[:, 1:] == pytest.approx(0.0)

    def test_current_density_is_the_curl_of_a_uniform_h(self, tmp_path, solves):
        paths = write_field_dumps([self.dump("Jf", 1e9)], tmp_path, verbose=False)

        magnitude = pv.read(paths[-2]).point_data["RotH-Field"]

        assert magnitude == pytest.approx(0.0, abs=1e-6)

    def test_a_repeated_request_is_written_once(self, tmp_path, solves):
        paths = write_field_dumps(
            [self.dump("Ef", 1e9), self.dump("Ef", 1e9)], tmp_path, verbose=False
        )

        assert len(paths) == 46
        assert paths[:23] == paths[23:]

    def test_reports_each_dump_when_verbose(self, tmp_path, solves, capsys):
        write_field_dumps([self.dump("Ef", 1e9)], tmp_path)

        assert "FEM field dump" in capsys.readouterr().out

    def test_prefers_the_domain_bbox_over_the_mesh_bbox(self, tmp_path, solves):
        meta_path = tmp_path / "fem_mesh.json"
        meta = json.loads(meta_path.read_text())
        meta["domain_bbox"] = [0, 0, 0, 0.12, 1, 1]
        meta_path.write_text(json.dumps(meta))

        (frame_path, *_rest) = write_field_dumps(
            [self.dump("Ef", 1e9)], tmp_path, verbose=False
        )

        assert pv.read(frame_path).x == pytest.approx([0.1])


# ---------------------------------------------------------------------
# SimTools.add_field_dump under the FEM backend
# ---------------------------------------------------------------------
def fem_line(fr4, **extra):
    from simpleEMS.microstrip_line import MicrostripLineParams

    return MicrostripLineParams(
        min_freq=2e9,
        max_freq=3e9,
        target_freq=2.45e9,
        backend_engine="FEM",
        **fr4,
        **extra,
    )


def build_line(params):
    from simpleEMS.microstrip_line import MicrostripLine
    from simpleEMS.sim_tools import setup_simulation

    sim = setup_simulation(params)
    line = MicrostripLine(params, sim)
    line.build_microstrip_line()
    return line, sim


@pytest.mark.needs_csxcad
class TestAddFieldDumpUnderFem:
    def test_queues_the_dump_on_the_setup(self, fr4):
        from simpleEMS.sim_tools import DumpType, SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)

        SimTools.add_field_dump(
            sim, params, dump_type=DumpType.current_density_frequency
        )

        (dump,) = sim.FEM_field_dumps
        assert dump.prefix == "Jf"
        assert dump.freq == params.main_freq

    def test_box_spans_the_board_in_metres(self, fr4):
        from simpleEMS.sim_tools import SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)

        SimTools.add_field_dump(sim, params, dump_freq=2.4e9)

        (dump,) = sim.FEM_field_dumps
        copper_top = params.substrate_thickness_mm + params.copper_thickness_mm
        assert dump.freq == 2.4e9
        assert dump.box[2] == 0.0
        assert dump.box[5] == pytest.approx(copper_top * 1e-3)

    def test_samples_on_the_fdtd_mesh_lines_in_metres(self, fr4):
        from simpleEMS.sim_tools import SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)

        SimTools.add_field_dump(sim, params)

        (dump,) = sim.FEM_field_dumps
        grid = sim.CSX.GetGrid()
        for axis in range(3):
            assert dump.lines[axis] == pytest.approx(
                np.asarray(grid.GetLines(axis)) * 1e-3
            )

    def test_writes_nothing_into_the_geometry(self, fr4):
        from simpleEMS.sim_tools import SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)
        before = len(sim.CSX.GetAllProperties())

        SimTools.add_field_dump(sim, params)

        assert len(sim.CSX.GetAllProperties()) == before

    @pytest.mark.parametrize(
        "dump_type",
        ["efield_time", "current_frequency", "local_sar_frequency"],
    )
    def test_refuses_what_the_fem_backend_cannot_dump(self, fr4, dump_type):
        from simpleEMS.sim_tools import DumpType, SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)

        with pytest.raises(ValueError, match="cannot dump"):
            SimTools.add_field_dump(sim, params, dump_type=DumpType[dump_type])


@pytest.mark.slow
@pytest.mark.needs_csxcad
@pytest.mark.needs_getdp_bin
def test_run_simulation_writes_the_queued_dumps(fr4, tmp_path):
    """Coarse on purpose, as in test_solver_smoke.py: this proves the dumps are
    written and readable, not that the fields are accurate."""
    from simpleEMS.sim_tools import DumpType, SimTools

    params = fem_line(
        fr4,
        num_points=11,
        FEM_num_solve_points=4,
        FEM_air_pad_mm=3.0,
        FEM_elems_per_wavelength=6.0,
        FEM_min_layers=1,
    )
    _line, sim = build_line(params)
    SimTools.add_field_dump(sim, params, tmp_path)
    SimTools.add_field_dump(
        sim, params, tmp_path, dump_type=DumpType.current_density_frequency
    )

    SimTools.run_simulation(sim, tmp_path)

    dump_dir = tmp_path / "field_dump"
    efield = pv.read(dump_dir / "Ef_f=2450000000_abs.vtr")
    current = pv.read(dump_dir / "Jf_f=2450000000_abs.vtr")
    assert efield.n_points > 0
    assert np.all(np.isfinite(efield.point_data["E-Field"]))
    assert current.point_data["RotH-Field"].max() > 0
    assert len(list(dump_dir.glob("Ef_f=2450000000_p=*.vtr"))) == 21
