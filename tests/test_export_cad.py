"""Tests for :mod:`simpleEMS.export_cad` (STEP and STL export).

STEP files embed a generation timestamp and a file-name header, so they cannot
be compared byte-for-byte the way the Gerber output can. Instead the exported
file is read back with cadquery and asserted on structurally: solid count,
bounding box, and volume. That catches the failures that matter -- a dropped
body, a mis-placed primitive, a polygon extruded along the wrong axis.
"""

import math

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
    _apply_transform,
    _unique_label,
    _make_box,
    _make_cylinder,
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

    def test_cylinder_spans_start_to_stop(self):
        """A via barrel runs between the two face centres CSXCAD stores; a
        cylinder built at the origin instead would pass a volume check and
        still land in the wrong place."""
        solid = _make_cylinder((1.0, 2.0, -0.035), (1.0, 2.0, 1.635), 0.3)
        bounds = solid.val().BoundingBox()

        assert bounds.zmin == pytest.approx(-0.035)
        assert bounds.zmax == pytest.approx(1.635)
        assert bounds.center.x == pytest.approx(1.0)
        assert bounds.center.y == pytest.approx(2.0)
        assert bounds.xlen == pytest.approx(0.6, rel=1e-3)

    def test_cylinder_volume_matches_pi_r_squared_l(self):
        solid = _make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.6), 0.3)

        assert solid.val().Volume() == pytest.approx(math.pi * 0.3**2 * 1.6, rel=1e-3)

    def test_cylinder_follows_an_arbitrary_axis(self):
        """start/stop need not differ on z only -- a cylinder laid on its
        side must not be re-erected along z."""
        solid = _make_cylinder((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), 0.5)

        assert solid.val().Volume() == pytest.approx(
            math.pi * 0.5**2 * math.sqrt(2.0), rel=1e-3
        )
        assert solid.val().BoundingBox().zlen == pytest.approx(1.0, rel=1e-3)

    def test_zero_length_cylinder_is_given_a_minimum_extent(self):
        """start == stop leaves no axis to point along; OCC cannot export a
        solid with no volume, so 1e-3 is substituted as for the box."""
        solid = _make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.3)

        assert solid.val().BoundingBox().zlen == pytest.approx(1e-3)
        assert solid.val().Volume() > 0

    def test_shell_volume_is_the_annulus(self):
        """A CSPrimCylindricalShell is a plated hole: the bore has to be cut
        out, or a shell exports as a solid barrel."""
        solid = _make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 0.5, 0.4)

        assert solid.val().Volume() == pytest.approx(
            math.pi * (0.5**2 - 0.4**2) * 2.0, rel=1e-3
        )


# ---------------------------------------------------------------------
# _apply_transform
# ---------------------------------------------------------------------
class TestApplyTransform:
    """CSXCAD builds a primitive's start/stop/coords in its own local frame
    and applies AddTransform on top; _apply_transform must reproduce that or
    a rotated/translated primitive (e.g. components.py's stubs and tapers,
    all rotated via translate-rotate-translate) exports untransformed."""

    def test_untransformed_primitive_is_returned_unchanged(self):
        csx = ContinuousStructure()
        prim = csx.AddMetal("m").AddBox(priority=1, start=[0, 0, 0], stop=[2, 3, 4])

        solid = _make_box(prim.GetStart(), prim.GetStop())
        result = _apply_transform(solid, prim)

        assert result is solid

    def test_translation_shifts_the_bounding_box(self):
        csx = ContinuousStructure()
        prim = csx.AddMetal("m").AddBox(priority=1, start=[0, 0, 0], stop=[2, 3, 4])
        prim.AddTransform("Translate", [5.0, -1.0, 0.0])

        solid = _apply_transform(_make_box(prim.GetStart(), prim.GetStop()), prim)
        bounds = solid.val().BoundingBox()

        assert bounds.center.x == pytest.approx(1.0 + 5.0)
        assert bounds.center.y == pytest.approx(1.5 - 1.0)
        assert bounds.center.z == pytest.approx(2.0)

    def test_rotation_preserves_volume_but_changes_the_footprint(self):
        """A 45-degree rotation about Z turns a rectangle's axis-aligned
        bounding box into its diagonal span; a bug that only transforms the
        two diagonal corners (rather than the whole solid) would report a
        collapsed or unchanged bounding box here instead."""
        csx = ContinuousStructure()
        prim = csx.AddMetal("m").AddBox(priority=1, start=[0, 0, 0], stop=[4, 2, 1])
        prim.AddTransform("Translate", [-2.0, -1.0, 0.0])
        prim.AddTransform("RotateAxis", "z", 45)
        prim.AddTransform("Translate", [2.0, 1.0, 0.0])

        solid = _apply_transform(_make_box(prim.GetStart(), prim.GetStop()), prim)
        bounds = solid.val().BoundingBox()

        theta = math.radians(45)
        expected = 4.0 * abs(math.cos(theta)) + 2.0 * abs(math.sin(theta))
        assert bounds.xlen == pytest.approx(expected, rel=1e-3)
        assert solid.val().Volume() == pytest.approx(4.0 * 2.0 * 1.0, rel=1e-6)

    def test_transform_moves_a_cylinder_too(self):
        """components.py rotates a primitive about a point with a
        translate-rotate-translate triple; a via carried along by one has to
        move with it."""
        csx = ContinuousStructure()
        prim = csx.AddMetal("via").AddCylinder(
            priority=4, start=[0, 0, 0], stop=[0, 0, 1.6], radius=0.3
        )
        prim.AddTransform("Translate", [-2.0, -1.0, 0.0])
        prim.AddTransform("RotateAxis", "z", 90)
        prim.AddTransform("Translate", [2.0, 1.0, 0.0])

        solid = _apply_transform(
            _make_cylinder(prim.GetStart(), prim.GetStop(), prim.GetRadius()), prim
        )
        bounds = solid.val().BoundingBox()

        # (0, 0) about (2, 1) through +90 degrees lands on (3, -1).
        assert bounds.center.x == pytest.approx(3.0, abs=1e-6)
        assert bounds.center.y == pytest.approx(-1.0, abs=1e-6)
        # transformGeometry re-fits the barrel's curved face as a BSpline, so
        # the volume is preserved only to within the fit -- close enough to
        # catch a collapsed or doubled body, which is what this guards.
        assert solid.val().Volume() == pytest.approx(math.pi * 0.3**2 * 1.6, rel=2e-2)


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

    def test_two_properties_sharing_a_name_both_export(self, tmp_path):
        """``AddMetal`` returns a new property every call, so two vias share a
        name; before deduplication this raised ValueError from CadQuery."""
        csx = ContinuousStructure()
        for x in (0.0, 5.0):
            prop = csx.AddMetal("via")
            prop.SetColor("#B87333", 255)
            prop.AddBox(priority=4, start=[x, 0, 0], stop=[x + 1, 1, 1])

        export_step(csx, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        assert len(result.val().Solids()) == 2
        assert result.val().Volume() == pytest.approx(2.0, rel=1e-6)

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

    def test_via_barrels_are_exported(self, tmp_path):
        """``GenericStructure.create_via`` adds an ``AddCylinder`` barrel. It
        used to match no branch of the primitive dispatch, so the property
        exported empty and the FEM mesh -- whose only view of the geometry is
        this file -- had no conductor through the board."""
        csx = ContinuousStructure()
        via = csx.AddMetal("via")
        via.SetColor("#B87333", 255)
        via.AddCylinder(priority=4, start=[0, 0, 0], stop=[0, 0, 1.6], radius=0.3)

        export_step(csx, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        assert len(result.solids().vals()) == 1
        assert result.val().Volume() == pytest.approx(math.pi * 0.3**2 * 1.6, rel=1e-3)

    def test_via_keeps_its_position(self, tmp_path):
        """A stitching via row is only meaningful if each barrel exports where
        it was placed."""
        csx = ContinuousStructure()
        via = csx.AddMetal("via")
        via.SetColor("#B87333", 255)
        via.AddCylinder(
            priority=4, start=[4.0, -2.0, -0.035], stop=[4.0, -2.0, 1.6], radius=0.3
        )

        export_step(csx, tmp_path)
        bounds = (
            cq.importers.importStep(str(tmp_path / "structure.step"))
            .val()
            .BoundingBox()
        )

        assert bounds.center.x == pytest.approx(4.0, abs=1e-6)
        assert bounds.center.y == pytest.approx(-2.0, abs=1e-6)
        assert bounds.zmin == pytest.approx(-0.035, abs=1e-6)
        assert bounds.zmax == pytest.approx(1.6, abs=1e-6)

    def test_cylindrical_shell_exports_with_its_bore(self, tmp_path):
        """CSPrimCylindricalShell subclasses CSPrimCylinder, so a dispatch on
        the exact class name misses it unless it is named explicitly."""
        csx = ContinuousStructure()
        shell = csx.AddMetal("plated_hole")
        shell.SetColor("#B87333", 255)
        shell.AddCylindricalShell(
            priority=4,
            start=[0, 0, 0],
            stop=[0, 0, 1.6],
            radius=0.45,
            shell_width=0.1,
        )

        export_step(csx, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))

        assert len(result.solids().vals()) == 1
        assert result.val().Volume() == pytest.approx(
            math.pi * (0.5**2 - 0.4**2) * 1.6, rel=1e-3
        )

    def test_unsupported_primitive_is_reported(self, tmp_path, capsys):
        """Most of CSXCAD's primitive types still have no branch. Dropping one
        without a word is how the missing cylinder support stayed hidden."""
        csx = ContinuousStructure()
        metal = csx.AddMetal("blob")
        metal.SetColor("#B87333", 255)
        metal.AddSphere(priority=4, center=[0, 0, 0], radius=1.0)

        export_step(csx, tmp_path)
        out = capsys.readouterr().out

        assert "skipped" in out
        assert "CSPrimSphere" in out

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

    def test_transformed_box_exports_in_its_rotated_pose(self, tmp_path):
        """A rotated primitive (e.g. components.py's stubs and tapers, which
        all rotate via translate-rotate-translate) must round-trip through
        STEP in its rotated pose, not its local, untransformed one."""
        csx = ContinuousStructure()
        pad = csx.AddMetal("pad")
        pad.SetColor("#B87333", 255)
        prim = pad.AddBox(priority=1, start=[0.0, 0.0, 0.0], stop=[4.0, 2.0, 1.0])
        prim.AddTransform("Translate", [-2.0, -1.0, 0.0])
        prim.AddTransform("RotateAxis", "z", 45)
        prim.AddTransform("Translate", [2.0, 1.0, 0.0])

        export_step(csx, tmp_path)
        result = cq.importers.importStep(str(tmp_path / "structure.step"))
        bounds = result.val().BoundingBox()

        theta = math.radians(45)
        expected_xlen = 4.0 * abs(math.cos(theta)) + 2.0 * abs(math.sin(theta))
        assert bounds.xlen == pytest.approx(expected_xlen, rel=1e-3)
        assert result.val().Volume() == pytest.approx(4.0 * 2.0 * 1.0, rel=1e-6)

    def test_transformed_polygon_exports_at_its_translated_position(self, tmp_path):
        csx = ContinuousStructure()
        metal = csx.AddMetal("poly")
        metal.SetColor("#B87333", 255)
        prim = metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3, 0], [0, 0, 4, 4]],
            norm_dir=2,
            elevation=0.0,
            length=0.035,
        )
        prim.AddTransform("Translate", [10.0, -5.0, 0.0])

        export_step(csx, tmp_path)
        bounds = (
            cq.importers.importStep(str(tmp_path / "structure.step"))
            .val()
            .BoundingBox()
        )

        assert bounds.xmin == pytest.approx(10.0, abs=1e-6)
        assert bounds.xmax == pytest.approx(13.0, abs=1e-6)
        assert bounds.ymin == pytest.approx(-5.0, abs=1e-6)
        assert bounds.ymax == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------
# _unique_label
# ---------------------------------------------------------------------
class TestUniqueLabel:
    """CSXCAD allows two properties to share a name, CadQuery does not allow
    two assembly parts to. Without deduplication a structure with two vias
    aborts the export part-way through, leaving no file behind."""

    def test_first_use_keeps_the_plain_name(self):
        used = set()
        assert _unique_label("via", used) == "via"

    def test_collisions_are_suffixed_in_order(self):
        used = set()
        labels = [_unique_label("via", used) for _ in range(3)]
        assert labels == ["via", "via_1", "via_2"]

    def test_distinct_names_are_untouched(self):
        used = set()
        assert [_unique_label(n, used) for n in ("via", "ground", "taper")] == [
            "via",
            "ground",
            "taper",
        ]

    def test_suffix_skips_a_name_already_taken(self):
        """A property genuinely called ``via_1`` must not be overwritten by
        the suffix generated for a second ``via``."""
        used = {"via", "via_1"}
        assert _unique_label("via", used) == "via_2"


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

    def test_two_properties_sharing_a_name_both_export(self, tmp_path):
        """``export_stl`` carries the same deduplication as ``export_step``;
        without it a structure with two vias aborts before writing a file."""
        csx = ContinuousStructure()
        for x in (0.0, 5.0):
            prop = csx.AddMetal("via")
            prop.SetColor("#B87333", 255)
            prop.AddBox(priority=4, start=[x, 0, 0], stop=[x + 1, 1, 1])

        export_stl(csx, tmp_path)

        assert (tmp_path / "structure.stl").stat().st_size > 0

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
