"""Tests for the gmsh-side helpers in :mod:`simpleEMS.fem_geometry`.

The full STEP-to-mesh pipeline is exercised by ``test_fem_step.py``. These
tests pin the small helpers inside it on hand-built gmsh entities, so the
branches a working model never takes -- a port with nothing measured, a sheet
that misses the port -- still get checked. Nothing here meshes or solves.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

# gmsh dlopen()s X/GL libraries at import; on a bare host that raises OSError,
# which importorskip does not catch.
try:
    import gmsh
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

from simpleEMS import fem_geometry  # noqa: E402
from simpleEMS.fem_backend import FEMOptions, PortSpec, Problem  # noqa: E402

MM = 1e-3


@pytest.fixture
def session():
    """A fresh gmsh session, finalised afterwards whatever the test did."""
    fem_geometry._init()
    gmsh.model.add("test")
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


def wave_problem(*ports):
    return Problem(
        step_file="s.step",
        freqs=np.array([1e9]),
        ports=list(ports),
        options=FEMOptions(port_type="waveport"),
    )


class TestInit:
    def test_a_running_session_is_replaced_with_a_quiet_one(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        fem_geometry._init()
        try:
            assert gmsh.option.getNumber("General.Terminal") == 0
        finally:
            gmsh.finalize()


class TestStructureBbox:
    def test_ports_and_ignored_solids_do_not_count(self):
        originals = [
            ("pec", "trace", (0, 0, 0, 1, 2, 3), 6.0),
            ("port", "port_1", (-5, -5, -5, 9, 9, 9), 1.0),
            ("ignore", "widget", (-9, -9, -9, 9, 9, 9), 1.0),
            ("dielectric", "substrate", (-1, 0, -1, 1, 1, 0), 2.0),
        ]

        assert fem_geometry._structure_bbox(originals) == (-1, 0, -1, 1, 2, 3)

    def test_nothing_but_ports_measures_as_zeros(self):
        originals = [("port", "port_1", (1, 1, 1, 2, 2, 2), 1.0)]

        assert fem_geometry._structure_bbox(originals) == (0.0,) * 6


class TestWavePortFacesWithoutGeometry:
    def test_a_port_with_no_measured_solid_places_no_wall(self):
        problem = wave_problem(
            PortSpec("port_1", 1, kind="wave", prop_dir="y"),
            PortSpec("port_2", 2),
        )

        faces, planes = fem_geometry._wave_port_faces(
            problem, {}, [], (0, 0, 0, 1, 1, 1), 0.1
        )

        assert (faces, planes) == ({}, {})

    def test_building_a_port_with_no_measured_solid_is_refused(self):
        problem = wave_problem(PortSpec("port_1", 1, kind="wave", prop_dir="y"))

        with pytest.raises(RuntimeError, match="none of its solids"):
            fem_geometry._build_wave_port_sheets(
                problem, [], {}, (0, 0, 0, 1, 1, 1), (0, 0, 0, 1, 1, 1)
            )


class TestSnapPortToConductorSheets:
    """A port drawn to the bottom of a trace has to reach the sheet the trace
    became on its top face."""

    TRACE = (-1 * MM, -5 * MM, 1.6 * MM, 1 * MM, 5 * MM, 1.635 * MM)
    TOP_SHEET = [-1 * MM, -5 * MM, 1.635 * MM, 1 * MM, 5 * MM, 1.635 * MM]

    @staticmethod
    def port_sheet():
        """A port normal to y spanning z from the ground to the trace bottom."""
        tag = fem_geometry._wave_port_sheet(
            (-0.5 * MM, 0.0, 0.0, 0.5 * MM, 0.0, 1.6 * MM), "y"
        )
        gmsh.model.occ.synchronize()
        return tag

    def test_the_port_end_moves_onto_the_sheet(self, session):
        tag = self.port_sheet()

        moved = fem_geometry._snap_port_to_conductor_sheets(
            tag, "z", [(self.TRACE, self.TOP_SHEET)]
        )
        gmsh.model.occ.synchronize()

        # OCC hands the freed tag to the replacement, so compare extents.
        bounds = fem_geometry._vertex_bounds(2, moved)
        assert bounds[2] == pytest.approx(0.0, abs=1e-12)
        assert bounds[5] == pytest.approx(1.635 * MM)

    def test_a_sheet_standing_along_the_gap_is_passed_over(self, session):
        tag = self.port_sheet()
        wall = [-1 * MM, -5 * MM, 0.0, -1 * MM, 5 * MM, 1.635 * MM]

        kept = fem_geometry._snap_port_to_conductor_sheets(
            tag, "z", [(self.TRACE, wall)]
        )

        assert kept == tag

    def test_a_sheet_beside_the_port_is_passed_over(self, session):
        tag = self.port_sheet()
        beside = [5 * MM, -5 * MM, 1.635 * MM, 6 * MM, 5 * MM, 1.635 * MM]

        kept = fem_geometry._snap_port_to_conductor_sheets(
            tag, "z", [(self.TRACE, beside)]
        )

        assert kept == tag


class TestSizeField:
    def test_nothing_to_refine_leaves_gmsh_sizing_alone(self, session):
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)

        fem_geometry._apply_size_field(set(), 0.1, 1.0, 2.0)

        assert gmsh.option.getNumber("Mesh.MeshSizeFromPoints") == 1

    def test_a_single_field_drives_the_mesh_on_its_own(self, session):
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        faces = {tag for _dim, tag in gmsh.model.getBoundary([(3, box)])}

        fem_geometry._apply_size_field(faces, 0.1, 1.0, 2.0)

        assert gmsh.option.getNumber("Mesh.MeshSizeFromPoints") == 0
        assert list(gmsh.model.mesh.field.list()) == [1, 2]


class TestPhysicalGroups:
    def test_a_port_that_produced_no_faces_is_refused(self, session):
        """Every port has to end up as a tagged region, or the solver has no
        boundary to excite."""
        faces = fem_geometry._Faces(
            pec=set(),
            imped_by_sigma={},
            port_by_name={},
            all_port=set(),
            all_imped=set(),
            ground=set(),
            abc=set(),
            sym=set(),
        )
        problem = Problem(
            step_file="s.step",
            freqs=np.array([1e9]),
            ports=[PortSpec("port_1", 1)],
        )

        with pytest.raises(RuntimeError, match="produced no boundary faces"):
            fem_geometry._assign_physical_groups(problem, {}, [], [], faces, {})
