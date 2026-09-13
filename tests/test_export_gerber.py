"""Tests for :mod:`simpleEMS.export_gerber`.

Gerber output is plain deterministic text, which makes golden files the right
tool: they pin the header, the coordinate format, the X2 attributes and the
contour winding all at once. The golden structure is built explicitly here
rather than derived from a params object, so a change to a design formula
changes ``test_calc.py`` -- not this file.

CSXCAD has no layers, so the exporter infers them from Z; most of the tests
below pin that inference through :func:`infer_layers` directly rather than by
parsing Gerber text.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from CSXCAD import ContinuousStructure  # noqa: E402

from simpleEMS.export_gerber import (  # noqa: E402
    export_gerber,
    gerber_coord,
    infer_layers,
    primitive_box,
    primitive_polygon,
)


GOLDEN_DIR = Path(__file__).parent / "golden" / "gerber"
GOLDEN_FILES = [
    "layout-F_Cu.gtl",
    "layout-In1_Cu.g2",
    "layout-B_Cu.gbl",
    "layout-Edge_Cuts.gm1",
    "layout-PTH.drl",
]

H = 1.6  # substrate thickness
T = 0.035  # copper thickness


def add_box(csx, name, z0, z1, xy=(-1.0, -1.0, 1.0, 1.0)):
    metal = csx.AddMetal(name)
    metal.AddBox(priority=1, start=[xy[0], xy[1], z0], stop=[xy[2], xy[3], z1])
    return metal


@pytest.fixture
def golden_structure():
    """A fixed 3-layer board: top pad + polygon, inner plane with an antipad,
    bottom ground, one through via, and an FR-4 substrate."""
    csx = ContinuousStructure()

    substrate = csx.AddMaterial("substrate", epsilon=4.4)
    substrate.AddBox(priority=0, start=[-5, -5, 0], stop=[5, 5, H])

    add_box(csx, "pad", H, H + T, xy=(-1.0, -2.0, 1.0, 2.0))

    poly = csx.AddMetal("poly")
    poly.AddLinPoly(
        priority=1,
        points=[[0, 3, 3, 0], [0, 0, 4, 4]],
        norm_dir=2,
        elevation=H,
        length=T,
    )

    add_box(csx, "inner", H / 2, H / 2 + T, xy=(-4.0, -4.0, 4.0, 4.0))
    add_box(csx, "ground", -T, 0.0, xy=(-5.0, -5.0, 5.0, 5.0))

    via = csx.AddMetal("via")
    via.AddCylinder(priority=2, start=[2, 2, -T], stop=[2, 2, H + T], radius=0.25)

    antipad = csx.AddMaterial("via_antipad", epsilon=1)
    antipad.AddLinPoly(
        priority=3,
        points=[[1.5, 2.5, 2.5, 1.5], [1.5, 1.5, 2.5, 2.5]],
        norm_dir=2,
        elevation=H / 2,
        length=T,
    )

    return csx


def export(csx, directory, options=None):
    """Run the export and return ``{filename: contents}``."""
    paths = export_gerber(csx, directory, options if options is not None else {})
    return {path.name: path.read_text() for path in paths}


def gerber_files(files):
    return {name: text for name, text in files.items() if not name.endswith(".drl")}


# ---------------------------------------------------------------------
# gerber_coord
# ---------------------------------------------------------------------
class TestGerberCoord:
    def test_format_matches_the_4_5_declaration(self):
        """``%FSLAX45Y45*%``: leading zeros omitted, 5 decimal digits."""
        assert gerber_coord((1.0, 2.0)) == "X100000Y200000"

    def test_scales_millimetres_by_one_hundred_thousand(self):
        assert gerber_coord((1.5, 0.0)) == "X150000Y0"

    def test_negative_coordinates_carry_a_minus_sign(self):
        assert gerber_coord((-1.0, -2.5)) == "X-100000Y-250000"

    def test_zero_is_written_as_a_bare_zero(self):
        assert gerber_coord((0.0, 0.0)) == "X0Y0"

    def test_rounds_to_the_nearest_ten_nanometres(self):
        """Five decimal places of a millimetre; anything finer is rounded,
        not truncated."""
        assert gerber_coord((1.234567, 0.0)).startswith("X123457Y")

    def test_sub_resolution_values_collapse_to_zero(self):
        assert gerber_coord((1e-7, -1e-7)) == "X0Y0"

    def test_accepts_lists_as_well_as_tuples(self):
        """``primitive_box`` passes both lists and numpy arrays in."""
        assert gerber_coord([1.0, 2.0]) == gerber_coord((1.0, 2.0))

    def test_largest_coordinate_fits_four_integer_digits(self):
        assert gerber_coord((9999.0, -9999.0)) == "X999900000Y-999900000"


# ---------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------
class TestGoldenFiles:
    @pytest.mark.parametrize("name", GOLDEN_FILES, ids=GOLDEN_FILES)
    def test_export_matches_the_golden_file(self, golden_structure, tmp_path, name):
        """Byte-for-byte. Regenerate the goldens only with a deliberate format
        change -- Gerber is consumed by fab tooling that is unforgiving."""
        files = export(golden_structure, tmp_path)

        assert files[name] == (GOLDEN_DIR / name).read_text()

    def test_export_writes_exactly_the_golden_set(self, golden_structure, tmp_path):
        assert sorted(export(golden_structure, tmp_path)) == sorted(GOLDEN_FILES)

    def test_export_is_reproducible(self, golden_structure, tmp_path):
        first = export(golden_structure, tmp_path)
        second = export(golden_structure, tmp_path)

        assert first == second


# ---------------------------------------------------------------------
# Layer inference
# ---------------------------------------------------------------------
class TestInferLayers:
    @pytest.mark.parametrize(
        ("z_ranges", "expected"),
        [
            ([(H, H + T)], ["F_Cu"]),
            ([(H, H + T), (-T, 0.0)], ["F_Cu", "B_Cu"]),
            ([(-T, 0.0), (H / 2, H / 2 + T), (H, H + T)], ["F_Cu", "In1_Cu", "B_Cu"]),
            (
                [(0.0, T), (0.4, 0.4 + T), (0.8, 0.8 + T), (1.2, 1.2 + T)],
                ["F_Cu", "In1_Cu", "In2_Cu", "B_Cu"],
            ),
        ],
        ids=["one-layer", "two-layer", "three-layer", "four-layer"],
    )
    def test_layers_are_named_from_the_top_down(self, z_ranges, expected):
        csx = ContinuousStructure()
        for i, (z0, z1) in enumerate(z_ranges):
            add_box(csx, f"m{i}", z0, z1)

        layers = infer_layers(csx).layers

        assert [layer.name for layer in layers] == expected
        assert [layer.z_min for layer in layers] == sorted(
            (z0 for z0, _ in z_ranges), reverse=True
        )

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, [("gtl", "Copper,L1,Top")]),
            (2, [("gtl", "Copper,L1,Top"), ("gbl", "Copper,L2,Bot")]),
            (
                4,
                [
                    ("gtl", "Copper,L1,Top"),
                    ("g2", "Copper,L2,Inr"),
                    ("g3", "Copper,L3,Inr"),
                    ("gbl", "Copper,L4,Bot"),
                ],
            ),
        ],
        ids=["one-layer", "two-layer", "four-layer"],
    )
    def test_extensions_and_file_functions_follow_kicad(self, count, expected):
        csx = ContinuousStructure()
        for i in range(count):
            add_box(csx, f"m{i}", i * 0.5, i * 0.5 + T)

        layers = infer_layers(csx).layers

        assert [(layer.extension, layer.file_function) for layer in layers] == expected

    def test_a_zero_thickness_polygon_joins_the_copper_it_touches(self):
        """Also a regression test: ``AddPolygon`` builds a ``CSPrimPolygon``,
        which the old exporter misspelt as ``CSPrimPoly`` and dropped."""
        csx = ContinuousStructure()
        add_box(csx, "pad", H, H + T)
        sheet = csx.AddMetal("sheet")
        sheet.AddPolygon(
            priority=1, points=[[0, 3, 3, 0], [0, 0, 4, 4]], norm_dir=2, elevation=H
        )

        layers = infer_layers(csx).layers

        assert len(layers) == 1
        assert [name for name, _ in layers[0].regions] == ["pad", "sheet"]

    def test_primitives_of_one_property_can_land_on_different_layers(self):
        csx = ContinuousStructure()
        metal = add_box(csx, "split", H, H + T)
        metal.AddBox(priority=1, start=[-1, -1, -T], stop=[1, 1, 0])

        layers = infer_layers(csx).layers

        assert [name for name, _ in layers[0].regions] == ["split"]
        assert [name for name, _ in layers[1].regions] == ["split"]

    def test_regions_keep_the_property_order(self, golden_structure):
        top = infer_layers(golden_structure).layers[0]

        assert [name for name, _ in top.regions] == ["pad", "poly"]

    def test_a_transform_moves_the_primitive_to_another_layer(self):
        csx = ContinuousStructure()
        add_box(csx, "ground", -T, 0.0)
        moved = csx.AddMetal("moved")
        box = moved.AddBox(priority=1, start=[-1, -1, -T], stop=[1, 1, 0])
        box.AddTransform("Translate", [0, 0, H + T])

        layers = infer_layers(csx).layers

        assert [layer.name for layer in layers] == ["F_Cu", "B_Cu"]
        assert [name for name, _ in layers[0].regions] == ["moved"]

    @pytest.mark.parametrize(
        ("start", "stop"),
        [
            ([4, -1, 0], [4, 1, H]),  # zero-thickness YZ sheet
            ([4, -0.1, 0], [4.2, 0.1, H]),  # narrow wall
        ],
        ids=["side-sheet", "wall"],
    )
    def test_vertical_metal_is_skipped(self, start, stop, capsys):
        csx = ContinuousStructure()
        add_box(csx, "pad", H, H + T)
        wall = csx.AddMetal("wall")
        wall.AddBox(priority=1, start=start, stop=stop)

        layers = infer_layers(csx).layers

        assert len(layers) == 1
        assert "vertical geometry" in capsys.readouterr().out

    def test_polygon_with_non_z_normal_is_skipped(self, capsys):
        """Gerber is a 2D format; a side-facing polygon has no XY footprint
        and must not be flattened into one."""
        csx = ContinuousStructure()
        side = csx.AddMetal("side")
        side.AddLinPoly(
            priority=1,
            points=[[0, 3, 3], [0, 0, 4]],
            norm_dir=0,
            elevation=0.0,
            length=T,
        )

        layout = infer_layers(csx)

        assert layout.layers == []
        assert "no XY footprint" in capsys.readouterr().out

    def test_ignore_removes_the_property_before_inference(self):
        csx = ContinuousStructure()
        add_box(csx, "pad", H, H + T)
        add_box(csx, "ground", -T, 0.0)

        layers = infer_layers(csx, ignore=["ground"]).layers

        assert [layer.name for layer in layers] == ["F_Cu"]

    def test_empty_structure_has_no_layers_or_outline(self):
        layout = infer_layers(ContinuousStructure())

        assert layout.layers == []
        assert layout.drills == []
        assert layout.outline is None


class TestDrills:
    def test_a_z_axis_cylinder_is_a_drill_not_a_layer(self, golden_structure):
        layout = infer_layers(golden_structure)

        assert len(layout.layers) == 3
        assert len(layout.drills) == 1
        drill = layout.drills[0]
        assert (drill.x, drill.y) == (2.0, 2.0)
        assert drill.diameter == pytest.approx(0.5)
        assert drill.span == (1, 3)

    def test_a_blind_via_spans_only_the_layers_it_touches(self):
        csx = ContinuousStructure()
        add_box(csx, "top", H, H + T)
        add_box(csx, "inner", H / 2, H / 2 + T)
        add_box(csx, "bottom", -T, 0.0)
        blind = csx.AddMetal("blind")
        blind.AddCylinder(
            priority=2, start=[0, 0, H / 2], stop=[0, 0, H + T], radius=0.1
        )

        assert infer_layers(csx).drills[0].span == (1, 2)

    def test_a_cylinder_off_the_z_axis_is_skipped(self, capsys):
        csx = ContinuousStructure()
        add_box(csx, "top", H, H + T)
        rod = csx.AddMetal("rod")
        rod.AddCylinder(priority=2, start=[0, 0, H], stop=[3, 0, H], radius=0.1)

        layout = infer_layers(csx)

        assert layout.drills == []
        assert "not along the Z axis" in capsys.readouterr().out

    def test_through_and_blind_vias_go_to_separate_files(self, tmp_path):
        csx = ContinuousStructure()
        add_box(csx, "top", H, H + T)
        add_box(csx, "inner", H / 2, H / 2 + T)
        add_box(csx, "bottom", -T, 0.0)
        via = csx.AddMetal("via")
        via.AddCylinder(priority=2, start=[0, 0, -T], stop=[0, 0, H + T], radius=0.2)
        via.AddCylinder(priority=2, start=[1, 0, H / 2], stop=[1, 0, H], radius=0.1)

        files = export(csx, tmp_path)

        assert "layout-PTH.drl" in files
        assert "; #@! TF.FileFunction,Plated,1,3,PTH" in files["layout-PTH.drl"]
        assert "; #@! TF.FileFunction,Plated,1,2,PTH" in files["layout-PTH-L1-L2.drl"]

    def test_holes_are_grouped_by_tool_diameter(self, tmp_path):
        csx = ContinuousStructure()
        add_box(csx, "top", H, H + T)
        add_box(csx, "bottom", -T, 0.0)
        via = csx.AddMetal("via")
        for x, radius in [(0, 0.2), (1, 0.4), (2, 0.2)]:
            via.AddCylinder(
                priority=2, start=[x, 0, -T], stop=[x, 0, H + T], radius=radius
            )

        drill = export(csx, tmp_path)["layout-PTH.drl"].splitlines()

        assert "T1C0.400" in drill
        assert "T2C0.800" in drill
        tool_1 = drill.index("T1", drill.index("%"))
        tool_2 = drill.index("T2", tool_1)
        assert drill[tool_1 + 1 : tool_2] == ["X0.0000Y0.0000", "X2.0000Y0.0000"]

    def test_no_drill_file_without_vias(self, tmp_path):
        csx = ContinuousStructure()
        add_box(csx, "top", H, H + T)

        assert not any(name.endswith(".drl") for name in export(csx, tmp_path))


class TestClearances:
    def test_an_antipad_inside_a_copper_layer_is_a_clearance(self, golden_structure):
        inner = infer_layers(golden_structure).layers[1]

        assert [name for name, _ in inner.clearances] == ["via_antipad"]

    def test_the_substrate_is_not_a_clearance(self, golden_structure):
        layers = infer_layers(golden_structure).layers

        for layer in layers:
            assert "substrate" not in [name for name, _ in layer.clearances]

    def test_clearances_are_written_in_clear_polarity_after_the_copper(
        self, golden_structure, tmp_path
    ):
        """``%LPC`` only erases what was drawn before it."""
        text = export(golden_structure, tmp_path)["layout-In1_Cu.g2"]

        assert text.index("%LNinner*%") < text.index("%LPC*%")
        assert text.index("%LPC*%") < text.index("%LNvia_antipad*%")
        assert text.rstrip().endswith("%LPD*%\nM02*")


class TestOutline:
    def test_outline_is_the_substrate_footprint(self, golden_structure):
        assert infer_layers(golden_structure).outline == (-5, -5, 5, 5)

    def test_outline_falls_back_to_the_copper_without_a_substrate(self):
        csx = ContinuousStructure()
        add_box(csx, "a", H, H + T, xy=(-1.0, -2.0, 1.0, 2.0))
        add_box(csx, "b", -T, 0.0, xy=(0.0, 0.0, 3.0, 1.0))

        assert infer_layers(csx).outline == (-1, -2, 3, 2)


# ---------------------------------------------------------------------
# export_gerber
# ---------------------------------------------------------------------
class TestExportGerber:
    def test_returns_the_paths_it_wrote(self, golden_structure, tmp_path):
        paths = export_gerber(golden_structure, tmp_path, {})

        assert all(path.is_file() and path.parent == tmp_path for path in paths)

    def test_prefix_names_the_files(self, golden_structure, tmp_path):
        paths = export_gerber(golden_structure, tmp_path, {}, prefix="board")

        assert all(path.name.startswith("board-") for path in paths)

    def test_every_gerber_file_has_the_kicad_style_header(
        self, golden_structure, tmp_path
    ):
        for text in gerber_files(export(golden_structure, tmp_path)).values():
            assert text.startswith("G04 gerber RS274X-file exported by simpleEMS*")
            assert "%FSLAX45Y45*%" in text
            assert "%MOMM*%" in text
            assert "%TF.FileFunction," in text

    def test_every_gerber_file_ends_with_the_end_of_file_marker(
        self, golden_structure, tmp_path
    ):
        for text in gerber_files(export(golden_structure, tmp_path)).values():
            assert text.rstrip().endswith("M02*")

    def test_dielectrics_are_not_copper(self, golden_structure, tmp_path):
        """Only copper is fabricated; a substrate emitted as a copper pour
        would short the whole board."""
        for text in gerber_files(export(golden_structure, tmp_path)).values():
            assert "%LNsubstrate*%" not in text

    def test_ground_is_exported_on_the_bottom_layer(self, golden_structure, tmp_path):
        files = export(golden_structure, tmp_path)

        assert "%LNground*%" in files["layout-B_Cu.gbl"]
        assert "%LNground*%" not in files["layout-F_Cu.gtl"]

    def test_ignore_option_skips_named_properties(self, golden_structure, tmp_path):
        files = export(golden_structure, tmp_path, {"ignore": ["pad"]})

        assert "%LNpad*%" not in files["layout-F_Cu.gtl"]
        assert "%LNpoly*%" in files["layout-F_Cu.gtl"]

    def test_missing_ignore_key_is_treated_as_empty(self, golden_structure, tmp_path):
        assert export(golden_structure, tmp_path, {}) == export(
            golden_structure, tmp_path, {"ignore": []}
        )

    def test_empty_structure_writes_nothing(self, tmp_path):
        assert export_gerber(ContinuousStructure(), tmp_path, {}) == []

    def test_every_region_is_opened_and_closed(self, golden_structure, tmp_path):
        """``G36``/``G37`` bracket each filled contour; an unbalanced pair
        makes the rest of the file render as one giant pour."""
        for text in gerber_files(export(golden_structure, tmp_path)).values():
            assert text.count("G36*") == text.count("G37*")

    def test_every_contour_starts_with_a_move(self, golden_structure, tmp_path):
        """The first coordinate after ``G36`` must be a D02 move; a D01 there
        would draw a line in from wherever the head happened to be."""
        for text in gerber_files(export(golden_structure, tmp_path)).values():
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if line == "G36*":
                    assert lines[index + 1].endswith("D02*")


# ---------------------------------------------------------------------
# Primitive writers
# ---------------------------------------------------------------------
class TestPrimitiveWriters:
    def test_box_emits_five_points_closing_the_rectangle(self, tmp_path):
        """Four corners plus an explicit return to the start."""
        csx = ContinuousStructure()
        metal = csx.AddMetal("m")
        metal.AddBox(priority=1, start=[0.0, 0.0, 0.0], stop=[2.0, 3.0, 0.035])
        box = csx.GetAllPrimitives()[0]

        path = tmp_path / "out.gbr"
        with open(path, "w") as handle:
            primitive_box(handle, box)
        content = path.read_text()

        assert content.count("D02*") == 1
        assert content.count("D01*") == 4
        first = content.splitlines()[1]
        last = content.splitlines()[-2]
        assert first[:-4] == last[:-4]

    def test_box_traces_the_corners_in_order(self, tmp_path):
        csx = ContinuousStructure()
        metal = csx.AddMetal("m")
        metal.AddBox(priority=1, start=[0.0, 0.0, 0.0], stop=[2.0, 3.0, 0.035])
        box = csx.GetAllPrimitives()[0]

        path = tmp_path / "out.gbr"
        with open(path, "w") as handle:
            primitive_box(handle, box)
        body = path.read_text().splitlines()[1:-1]

        assert body[0] == "X0Y0D02*"
        assert body[1] == "X200000Y0D01*"
        assert body[2] == "X200000Y300000D01*"
        assert body[3] == "X0Y300000D01*"

    def test_polygon_with_non_z_normal_is_not_written(self, tmp_path, capsys):
        csx = ContinuousStructure()
        metal = csx.AddMetal("side")
        metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3], [0, 0, 4]],
            norm_dir=0,
            elevation=0.0,
            length=0.035,
        )
        poly = csx.GetAllPrimitives()[0]

        path = tmp_path / "out.gbr"
        with open(path, "w") as handle:
            primitive_polygon(handle, poly)

        assert path.read_text() == ""
        assert "normal direction is not +Z" in capsys.readouterr().out

    def test_polygon_closes_back_to_its_first_vertex(self, tmp_path):
        csx = ContinuousStructure()
        metal = csx.AddMetal("m")
        metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3, 0], [0, 0, 4, 4]],
            norm_dir=2,
            elevation=0.0,
            length=0.035,
        )
        poly = csx.GetAllPrimitives()[0]

        path = tmp_path / "out.gbr"
        with open(path, "w") as handle:
            primitive_polygon(handle, poly)
        body = path.read_text().splitlines()

        assert body[1][:-4] == body[-2][:-4]

    def test_polygon_emits_every_vertex(self, tmp_path):
        csx = ContinuousStructure()
        metal = csx.AddMetal("m")
        metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3, 0, -1], [0, 0, 4, 4, 2]],
            norm_dir=2,
            elevation=0.0,
            length=0.035,
        )
        poly = csx.GetAllPrimitives()[0]

        path = tmp_path / "out.gbr"
        with open(path, "w") as handle:
            primitive_polygon(handle, poly)
        content = path.read_text()

        # 5 vertices -> 1 move + 4 draws + 1 closing draw
        assert content.count("D02*") == 1
        assert content.count("D01*") == 5
