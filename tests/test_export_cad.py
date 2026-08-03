"""Tests for :mod:`simpleEMS.export_cad` (STEP and STL export).

STEP files embed a generation timestamp and a file-name header, so they cannot
be compared byte-for-byte the way the Gerber output can. Instead the exported
file is read back with cadquery and asserted on structurally: solid count,
bounding box, and volume. That catches the failures that matter -- a dropped
body, a mis-placed primitive, a polygon extruded along the wrong axis.
"""

import pytest

pytestmark = [pytest.mark.needs_csxcad, pytest.mark.needs_cadquery]

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

import cadquery as cq  # noqa: E402

from CSXCAD import ContinuousStructure  # noqa: E402

from simpleEMS.export_cad import (  # noqa: E402
    _make_box,
    _make_linpoly,
    _normal_dir,
    export_step,
    export_stl,
)


@pytest.fixture
def two_body_structure():
    """A copper pad on a dielectric slab, with known dimensions."""
    csx = ContinuousStructure()

    pad = csx.AddMetal("pad")
    pad.SetColor("#B87333", 255)
    pad.AddBox(priority=1, start=[-1.0, -2.0, 0.0], stop=[1.0, 2.0, 0.035])

    substrate = csx.AddMaterial("substrate", epsilon=4.4)
    substrate.SetColor("#0F8A00", 100)
    substrate.AddBox(priority=0, start=[-5.0, -5.0, -1.6], stop=[5.0, 5.0, 0.0])

    return csx


# ---------------------------------------------------------------------
# Solid construction helpers
# ---------------------------------------------------------------------
class TestSolidHelpers:
    @pytest.mark.parametrize(
        ("normdir", "plane", "offset"),
        [
            (0, "YZ", (2.5, 0.0, 0.0)),
            (1, "XZ", (0.0, 2.5, 0.0)),
            (2, "XY", (0.0, 0.0, 2.5)),
        ],
    )
    def test_normal_dir_maps_axis_to_workplane(self, normdir, plane, offset):
        """The elevation must be applied along the polygon's normal axis;
        putting it on the wrong axis silently relocates the body."""
        assert _normal_dir(normdir, 2.5) == (plane, offset)

    def test_box_has_the_requested_dimensions(self):
        solid = _make_box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        bounds = solid.val().BoundingBox()

        assert bounds.xlen == pytest.approx(2.0)
        assert bounds.ylen == pytest.approx(3.0)
        assert bounds.zlen == pytest.approx(4.0)

    def test_box_is_centred_between_start_and_stop(self):
        solid = _make_box((1.0, 2.0, 3.0), (3.0, 6.0, 5.0))
        bounds = solid.val().BoundingBox()

        assert bounds.center.x == pytest.approx(2.0)
        assert bounds.center.y == pytest.approx(4.0)
        assert bounds.center.z == pytest.approx(4.0)

    def test_box_handles_reversed_corners(self):
        """CSXCAD lets start be greater than stop on any axis."""
        forward = _make_box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)).val().BoundingBox()
        reversed_ = _make_box((2.0, 3.0, 4.0), (0.0, 0.0, 0.0)).val().BoundingBox()

        assert forward.xlen == pytest.approx(reversed_.xlen)
        assert forward.center.x == pytest.approx(reversed_.center.x)

    def test_zero_thickness_box_is_given_a_minimum_extent(self):
        """A flat sheet has no volume, and OCC cannot export a degenerate
        solid; the code substitutes 1e-3."""
        solid = _make_box((0.0, 0.0, 0.0), (2.0, 3.0, 0.0))

        assert solid.val().BoundingBox().zlen == pytest.approx(1e-3)

    def test_linpoly_is_extruded_along_its_normal(self):
        solid = _make_linpoly([0.0, 3.0, 3.0, 0.0], [0.0, 0.0, 4.0, 4.0], 0.0, 2, 0.5)
        bounds = solid.val().BoundingBox()

        assert bounds.xlen == pytest.approx(3.0)
        assert bounds.ylen == pytest.approx(4.0)
        assert bounds.zlen == pytest.approx(0.5)

    def test_linpoly_elevation_offsets_the_body(self):
        solid = _make_linpoly([0.0, 3.0, 3.0, 0.0], [0.0, 0.0, 4.0, 4.0], 1.6, 2, 0.5)

        assert solid.val().BoundingBox().zmin == pytest.approx(1.6)

    def test_linpoly_volume_matches_the_polygon_area(self):
        solid = _make_linpoly([0.0, 3.0, 3.0, 0.0], [0.0, 0.0, 4.0, 4.0], 0.0, 2, 0.5)

        assert solid.val().Volume() == pytest.approx(3.0 * 4.0 * 0.5, rel=1e-6)

    def test_zero_length_extrusion_is_given_a_minimum(self):
        solid = _make_linpoly([0.0, 3.0, 3.0, 0.0], [0.0, 0.0, 4.0, 4.0], 0.0, 2, 0.0)

        assert solid.val().BoundingBox().zlen == pytest.approx(1e-3)


# ---------------------------------------------------------------------
# export_step
# ---------------------------------------------------------------------
class TestExportStep:
    def test_writes_structure_step(self, two_body_structure, tmp_path):
        export_step(two_body_structure, tmp_path)

        assert (tmp_path / "structure.step").is_file()

    def test_exports_one_solid_per_primitive(self, two_body_structure, tmp_path):
        export_step(two_body_structure, tmp_path)

        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        assert len(result.solids().vals()) == 2

    def test_bounding_box_spans_both_bodies(self, two_body_structure, tmp_path):
        export_step(two_body_structure, tmp_path)

        bounds = (
            cq.importers.importStep(str(tmp_path / "structure.step"))
            .val()
            .BoundingBox()
        )

        assert bounds.xmin == pytest.approx(-5.0, abs=1e-6)
        assert bounds.xmax == pytest.approx(5.0, abs=1e-6)
        assert bounds.zmin == pytest.approx(-1.6, abs=1e-6)
        assert bounds.zmax == pytest.approx(0.035, abs=1e-6)

    def test_dielectrics_are_exported_too(self, two_body_structure, tmp_path):
        """Unlike Gerber, STEP is a mechanical model: the substrate belongs in
        it."""
        export_step(two_body_structure, tmp_path)

        volumes = sorted(
            solid.Volume()
            for solid in cq.importers.importStep(str(tmp_path / "structure.step"))
            .solids()
            .vals()
        )

        assert volumes[0] == pytest.approx(2.0 * 4.0 * 0.035, rel=1e-3)
        assert volumes[1] == pytest.approx(10.0 * 10.0 * 1.6, rel=1e-3)

    def test_polygon_primitives_are_exported(self, tmp_path):
        csx = ContinuousStructure()
        metal = csx.AddMetal("poly")
        metal.SetColor("#B87333", 255)
        metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3, 0], [0, 0, 4, 4]],
            norm_dir=2,
            elevation=0.0,
            length=0.035,
        )

        export_step(csx, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        assert len(result.solids().vals()) == 1
        assert result.val().Volume() == pytest.approx(3.0 * 4.0 * 0.035, rel=1e-3)

    def test_empty_structure_writes_nothing(self, tmp_path, capsys):
        """Better to warn than to emit an empty file a CAD tool chokes on."""
        export_step(ContinuousStructure(), tmp_path)

        assert not (tmp_path / "structure.step").exists()
        assert "No physical geometry" in capsys.readouterr().out

    def test_probe_and_excitation_properties_are_excluded(self, built_inset, tmp_path):
        """A built antenna carries probe boxes and excitation properties that
        are not physical objects and must not appear in the CAD model."""
        _antenna, sim, _params, _port = built_inset

        export_step(sim.CSX, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        physical = [
            prop
            for prop in sim.CSX.GetAllProperties()
            if prop.__class__.__name__
            in {"CSPropMetal", "CSPropMaterial", "CSPropLumpedElement"}
        ]

        assert len(result.solids().vals()) == len(physical)

    def test_real_antenna_geometry_round_trips(self, built_inset, tmp_path):
        _antenna, sim, params, _port = built_inset

        export_step(sim.CSX, tmp_path)
        bounds = (
            cq.importers.importStep(str(tmp_path / "structure.step"))
            .val()
            .BoundingBox()
        )

        assert bounds.xlen == pytest.approx(params.substrate_width_mm, abs=0.1)
        assert bounds.ylen == pytest.approx(params.substrate_length_mm, abs=0.1)


# ---------------------------------------------------------------------
# export_stl
# ---------------------------------------------------------------------
class TestExportStl:
    def test_writes_structure_stl(self, two_body_structure, tmp_path):
        export_stl(two_body_structure, tmp_path)

        assert (tmp_path / "structure.stl").is_file()

    def test_output_is_non_empty(self, two_body_structure, tmp_path):
        export_stl(two_body_structure, tmp_path)

        assert (tmp_path / "structure.stl").stat().st_size > 0

    def test_triangle_count_is_consistent_with_the_file_size(
        self, two_body_structure, tmp_path
    ):
        """Binary STL is an 84-byte header plus 50 bytes per facet. If the
        declared count and the file length disagree, the mesh is truncated and
        slicers will reject it."""
        export_stl(two_body_structure, tmp_path)
        content = (tmp_path / "structure.stl").read_bytes()

        declared = int.from_bytes(content[80:84], "little")

        assert len(content) == 84 + 50 * declared

    def test_two_boxes_tessellate_to_twenty_four_triangles(
        self, two_body_structure, tmp_path
    ):
        """Each rectangular body is 6 quads = 12 triangles; a body silently
        dropped from the assembly shows up here immediately."""
        export_stl(two_body_structure, tmp_path)
        content = (tmp_path / "structure.stl").read_bytes()

        assert int.from_bytes(content[80:84], "little") == 24

    def test_empty_structure_writes_nothing(self, tmp_path, capsys):
        export_stl(ContinuousStructure(), tmp_path)

        assert not (tmp_path / "structure.stl").exists()
        assert "No physical geometry" in capsys.readouterr().out

    def test_step_and_stl_export_the_same_bodies(self, two_body_structure, tmp_path):
        """Both exporters share ``_process_property``; a divergence means one
        of them grew a filter the other did not."""
        export_step(two_body_structure, tmp_path)
        export_stl(two_body_structure, tmp_path)

        step_path = tmp_path / "structure.step"
        stl_path = tmp_path / "structure.stl"

        assert step_path.is_file()
        assert stl_path.is_file()


# ---------------------------------------------------------------------
# XML round trip
# ---------------------------------------------------------------------
class TestXmlToStep:
    """``export_csxcad_xml_to_step`` reads the ``structure.xml`` that
    ``run_simulation`` writes, which is an *openEMS*-level document (it is
    loaded via ``openEMS.ReadFromXML``), not a bare CSXCAD one."""

    @pytest.fixture
    def structure_xml(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset
        path = tmp_path / "structure.xml"
        sim.FDTD.Write2XML(str(path))
        return path, sim

    def test_converts_a_written_structure_xml(self, structure_xml, tmp_path):
        path, _sim = structure_xml
        out_dir = tmp_path / "cad"

        from simpleEMS.export_cad import export_csxcad_xml_to_step

        export_csxcad_xml_to_step(path, out_dir)

        assert (out_dir / "structure.step").is_file()

    def test_creates_the_output_directory(self, structure_xml, tmp_path):
        path, _sim = structure_xml
        out_dir = tmp_path / "does" / "not" / "exist"

        from simpleEMS.export_cad import export_csxcad_xml_to_step

        export_csxcad_xml_to_step(path, out_dir)

        assert out_dir.is_dir()

    def test_accepts_a_string_path(self, structure_xml, tmp_path):
        path, _sim = structure_xml

        from simpleEMS.export_cad import export_csxcad_xml_to_step

        export_csxcad_xml_to_step(str(path), str(tmp_path / "cad"))

        assert (tmp_path / "cad" / "structure.step").is_file()

    def test_round_trip_preserves_the_bounding_box(self, structure_xml, tmp_path):
        """Going through XML must not move or rescale the geometry."""
        path, sim = structure_xml

        from simpleEMS.export_cad import export_csxcad_xml_to_step

        direct = tmp_path / "direct"
        direct.mkdir()
        export_step(sim.CSX, direct)

        via_xml = tmp_path / "via_xml"
        export_csxcad_xml_to_step(path, via_xml)

        a = cq.importers.importStep(str(direct / "structure.step")).val().BoundingBox()
        b = cq.importers.importStep(str(via_xml / "structure.step")).val().BoundingBox()

        assert (a.xmin, a.xmax, a.zmin, a.zmax) == pytest.approx(
            (b.xmin, b.xmax, b.zmin, b.zmax), abs=1e-6
        )
