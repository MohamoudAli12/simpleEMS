"""Tests for the FEM field dumps (``simpleEMS.fem_field_dump``).

The surface-current and box-filter tests run on hand-built tetrahedra, so they
need neither GetDP nor CSXCAD. The end-to-end test solves a coarse microstrip
line and checks the files ``SimTools.add_field_dump`` asks for appear.
"""

import numpy as np
import pytest

# gmsh dlopen()s X/GL libraries at import; on a bare host that raises OSError,
# which importorskip does not catch.
try:
    import gmsh  # noqa: F401
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

from simpleEMS import fem_field_dump  # noqa: E402
from simpleEMS.fem_field_dump import (  # noqa: E402
    FieldDumpRequest,
    _surface_current,
    _volume_grid,
    _VolumeFields,
)
from simpleEMS.fem_materials import AIR, PML  # noqa: E402

# A sheet triangle in the z = 0 plane, with one tetrahedron above it and one
# below. Points: 0-2 the triangle, 3 the apex above, 4 the apex below.
POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.2, 0.2, 1.0],
        [0.2, 0.2, -1.0],
    ]
)
ABOVE, BELOW = [0, 1, 2, 3], [4, 2, 1, 0]
SHEET = np.array([[0, 1, 2]])


def uniform_h(*fields):
    """``(n_tets, 4, 3)`` complex H, constant over each tetrahedron."""
    return np.array([[field] * 4 for field in fields], dtype=complex)


class TestSurfaceCurrent:
    def test_two_sided_sheet_carries_the_jump_in_h(self):
        """``n x H`` from each side: z x x above plus (-z) x (-x) below."""
        tets = np.array([ABOVE, BELOW])
        hfield = uniform_h([1, 0, 0], [-1, 0, 0])

        current = _surface_current(POINTS, tets, hfield, SHEET)

        assert current[0] == pytest.approx([0, 2, 0])

    def test_equal_h_either_side_carries_no_current(self):
        tets = np.array([ABOVE, BELOW])
        hfield = uniform_h([1, 0, 0], [1, 0, 0])

        current = _surface_current(POINTS, tets, hfield, SHEET)

        assert current[0] == pytest.approx([0, 0, 0])

    def test_one_sided_sheet_is_n_cross_h(self):
        tets = np.array([ABOVE])
        hfield = uniform_h([1, 0, 0])

        current = _surface_current(POINTS, tets, hfield, SHEET)

        assert current[0] == pytest.approx([0, 1, 0])

    @pytest.mark.parametrize(
        "sheet",
        [[0, 1, 2], [2, 1, 0], [1, 2, 0]],
        ids=["counterclockwise", "clockwise", "rotated"],
    )
    def test_node_order_does_not_flip_the_current(self, sheet):
        tets = np.array([ABOVE, BELOW])
        hfield = uniform_h([1, 0, 0], [-1, 0, 0])

        current = _surface_current(POINTS, tets, hfield, np.array([sheet]))

        assert current[0] == pytest.approx([0, 2, 0])

    def test_keeps_the_phase(self):
        tets = np.array([ABOVE])
        hfield = uniform_h([1j, 0, 0])

        current = _surface_current(POINTS, tets, hfield, SHEET)

        assert current[0] == pytest.approx([0, 1j, 0])

    def test_a_sheet_no_tetrahedron_touches_gets_zero(self):
        tets = np.array([ABOVE])
        hfield = uniform_h([1, 0, 0])
        stray = np.array([[0, 1, 4]])

        current = _surface_current(POINTS, tets, hfield, stray)

        assert current[0] == pytest.approx([0, 0, 0])


class TestVolumeGrid:
    @pytest.fixture
    def fields(self):
        """The two tetrahedra, the lower one in the given region."""

        def make(lower_region=AIR):
            return _VolumeFields(
                node_tags=np.arange(1, 6),
                points=POINTS,
                tets=np.array([ABOVE, BELOW]),
                regions=np.array([AIR, lower_region]),
                efield=uniform_h([1, 0, 0], [0, 2, 0]),
                hfield=uniform_h([0, 0, 3], [0, 0, 4]),
            )

        return make

    def test_no_box_keeps_the_whole_domain(self, fields):
        grid = _volume_grid(fields(), "E", None)

        assert grid.n_cells == 2

    def test_box_keeps_the_tetrahedra_centred_inside(self, fields):
        grid = _volume_grid(fields(), "E", (-1, -1, 0, 1, 1, 1))

        assert grid.n_cells == 1
        assert grid.point_data["E_magnitude"] == pytest.approx(1.0)

    def test_the_pml_is_left_out(self, fields):
        grid = _volume_grid(fields(lower_region=PML), "H", None)

        assert grid.n_cells == 1
        assert grid.point_data["H_real"][:, 2] == pytest.approx(3.0)

    def test_shared_nodes_average_the_tetrahedra(self, fields):
        grid = _volume_grid(fields(), "E", None)
        sheet_node = np.flatnonzero(np.all(grid.points == POINTS[0], axis=1))[0]

        assert grid.point_data["E_real"][sheet_node] == pytest.approx([0.5, 1, 0])

    def test_an_empty_box_is_refused(self, fields):
        with pytest.raises(ValueError, match="contains no mesh elements"):
            _volume_grid(fields(), "E", (5, 5, 5, 6, 6, 6))


class TestFieldDumpRequest:
    @pytest.mark.parametrize("prefix", ["Ef", "Hf", "Jf"])
    def test_accepts_the_frequency_domain_prefixes(self, prefix):
        assert FieldDumpRequest(prefix=prefix, freq=1e9).prefix == prefix

    @pytest.mark.parametrize("prefix", ["Et", "If", "SAR_f"])
    def test_refuses_what_the_fem_backend_cannot_dump(self, prefix):
        with pytest.raises(ValueError, match="prefix must be one of"):
            FieldDumpRequest(prefix=prefix, freq=1e9)


def test_write_field_dumps_needs_a_simulation_first(tmp_path):
    with pytest.raises(RuntimeError, match="run the simulation first"):
        fem_field_dump.write_field_dumps([FieldDumpRequest("Ef", 1e9)], tmp_path)


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

        (request,) = sim.FEM_field_dumps
        assert request.prefix == "Jf"
        assert request.freq == params.main_freq

    def test_box_spans_the_board_in_metres(self, fr4):
        from simpleEMS.sim_tools import SimTools

        params = fem_line(fr4)
        _line, sim = build_line(params)

        SimTools.add_field_dump(sim, params, dump_freq=2.4e9)

        (request,) = sim.FEM_field_dumps
        copper_top = params.substrate_thickness_mm + params.copper_thickness_mm
        assert request.freq == 2.4e9
        assert request.box[2] == 0.0
        assert request.box[5] == pytest.approx(copper_top * 1e-3)

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
    import pyvista as pv

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

    efield = pv.read(tmp_path / "field_dump" / "Ef_2.45GHz.vtu")
    current = pv.read(tmp_path / "field_dump" / "Jf_2.45GHz.vtu")
    assert efield.n_cells > 0
    assert current.n_cells > 0
    assert np.all(np.isfinite(efield.point_data["E_magnitude"]))
    assert current.cell_data["J_magnitude"].max() > 0
