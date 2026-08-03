"""Tests for :mod:`simpleEMS.export_gerber`.

Gerber output is plain deterministic text, which makes a golden file the right
tool: it pins the header, the coordinate format, the aperture definition, and
the contour winding all at once. The golden structure is built explicitly here
rather than derived from a params object, so a change to a design formula
changes ``test_calc.py`` -- not this file.
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
    primitive_box,
    primitive_polygon,
)


GOLDEN = Path(__file__).parent / "golden" / "gerber_layout.gbr"


@pytest.fixture
def golden_structure():
    """A fixed two-metal, one-dielectric structure with known coordinates."""
    csx = ContinuousStructure()

    pad = csx.AddMetal("pad")
    pad.AddBox(priority=1, start=[-1.0, -2.0, 0.0], stop=[1.0, 2.0, 0.035])

    poly = csx.AddMetal("poly")
    poly.AddLinPoly(
        priority=1,
        points=[[0, 3, 3, 0], [0, 0, 4, 4]],
        norm_dir=2,
        elevation=0.0,
        length=0.035,
    )

    substrate = csx.AddMaterial("substrate", epsilon=4.4)
    substrate.AddBox(priority=0, start=[-5, -5, -1], stop=[5, 5, 0])

    return csx


def export(csx, directory, options=None):
    """Run the export and return the file contents."""
    export_gerber(csx, directory, options if options is not None else {})
    return (directory / "gerber_layout.gbr").read_text()


# ---------------------------------------------------------------------
# gerber_coord
# ---------------------------------------------------------------------
class TestGerberCoord:
    def test_format_is_signed_and_zero_padded_to_thirteen(self):
        """``%FSLAX66Y66*%`` in the header declares 6 integer + 6 decimal
        digits; the coordinate strings must match that declaration."""
        result = gerber_coord((1.0, 2.0))

        assert result == "X+000001000000Y+000002000000"
        assert len(result) == 28

    def test_scales_millimetres_by_one_million(self):
        assert gerber_coord((1.5, 0.0)) == "X+000001500000Y+000000000000"

    def test_negative_coordinates_carry_a_minus_sign(self):
        assert gerber_coord((-1.0, -2.5)) == "X-000001000000Y-000002500000"

    def test_zero_is_positive_signed(self):
        assert gerber_coord((0.0, 0.0)) == "X+000000000000Y+000000000000"

    def test_rounds_to_the_nearest_nanometre(self):
        """Six decimal places of a millimetre; anything finer is rounded, not
        truncated."""
        assert gerber_coord((1.2345678, 0.0)).startswith("X+000001234568")

    def test_sub_nanometre_values_collapse_to_zero(self):
        assert gerber_coord((1e-9, 0.0)) == "X+000000000000Y+000000000000"

    def test_accepts_lists_as_well_as_tuples(self):
        """``primitive_box`` passes both lists and numpy arrays in."""
        assert gerber_coord([1.0, 2.0]) == gerber_coord((1.0, 2.0))

    def test_large_coordinates_do_not_overflow_the_field(self):
        result = gerber_coord((999.0, -999.0))

        assert result == "X+000999000000Y-000999000000"


# ---------------------------------------------------------------------
# Golden file
# ---------------------------------------------------------------------
class TestGoldenFile:
    def test_export_matches_the_golden_file(self, golden_structure, tmp_path):
        """Byte-for-byte. Regenerate the golden only with a deliberate format
        change -- Gerber is consumed by fab tooling that is unforgiving."""
        assert export(golden_structure, tmp_path) == GOLDEN.read_text()

    def test_export_is_reproducible(self, golden_structure, tmp_path):
        first = export(golden_structure, tmp_path)
        second = export(golden_structure, tmp_path)

        assert first == second


# ---------------------------------------------------------------------
# export_gerber
# ---------------------------------------------------------------------
class TestExportGerber:
    def test_writes_the_expected_filename(self, golden_structure, tmp_path):
        export_gerber(golden_structure, tmp_path, {})

        assert (tmp_path / "gerber_layout.gbr").is_file()

    def test_file_starts_with_the_rs274x_header(self, golden_structure, tmp_path):
        content = export(golden_structure, tmp_path)

        assert content.startswith("G04 gerber RS274X-file exported by simpleEMS*")
        assert "%FSLAX66Y66*%" in content
        assert "%MOMM*%" in content

    def test_file_ends_with_the_end_of_file_marker(self, golden_structure, tmp_path):
        content = export(golden_structure, tmp_path)

        assert content.rstrip().endswith("M02*")

    def test_dielectrics_are_excluded(self, golden_structure, tmp_path):
        """Only copper is fabricated; a substrate emitted as a copper pour
        would short the whole board."""
        content = export(golden_structure, tmp_path)

        assert "%LNsubstrate*%" not in content

    def test_metals_are_included(self, golden_structure, tmp_path):
        content = export(golden_structure, tmp_path)

        assert "%LNpad*%" in content
        assert "%LNpoly*%" in content

    def test_ignore_option_skips_named_properties(self, golden_structure, tmp_path):
        content = export(golden_structure, tmp_path, {"ignore": ["pad"]})

        assert "%LNpad*%" not in content
        assert "%LNpoly*%" in content

    def test_ignore_can_skip_everything(self, golden_structure, tmp_path):
        content = export(golden_structure, tmp_path, {"ignore": ["pad", "poly"]})

        assert "G36*" not in content
        assert content.rstrip().endswith("M02*")

    def test_missing_ignore_key_is_treated_as_empty(self, golden_structure, tmp_path):
        assert export(golden_structure, tmp_path, {}) == export(
            golden_structure, tmp_path, {"ignore": []}
        )

    def test_empty_structure_still_writes_a_valid_file(self, tmp_path):
        content = export(ContinuousStructure(), tmp_path)

        assert content.startswith("G04 gerber")
        assert content.rstrip().endswith("M02*")

    def test_property_without_primitives_emits_a_layer_name_only(self, tmp_path):
        csx = ContinuousStructure()
        csx.AddMetal("empty_layer")

        content = export(csx, tmp_path)

        assert "%LNempty_layer*%" in content
        assert "G36*" not in content

    def test_every_region_is_opened_and_closed(self, golden_structure, tmp_path):
        """``G36``/``G37`` bracket each filled contour; an unbalanced pair
        makes the rest of the file render as one giant pour."""
        content = export(golden_structure, tmp_path)

        assert content.count("G36*") == content.count("G37*")

    def test_every_contour_starts_with_a_move(self, golden_structure, tmp_path):
        """The first coordinate after ``G36`` must be a D02 move; a D01 there
        would draw a line in from wherever the head happened to be."""
        content = export(golden_structure, tmp_path)
        lines = content.splitlines()

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

        assert body[0].startswith("X+000000000000Y+000000000000")
        assert body[1].startswith("X+000002000000Y+000000000000")
        assert body[2].startswith("X+000002000000Y+000003000000")
        assert body[3].startswith("X+000000000000Y+000003000000")

    def test_polygon_with_non_z_normal_is_skipped(self, tmp_path, capsys):
        """Gerber is a 2D format; a side-facing polygon has no XY footprint
        and must not be flattened into one."""
        csx = ContinuousStructure()
        metal = csx.AddMetal("side")
        metal.AddLinPoly(
            priority=1,
            points=[[0, 3, 3], [0, 0, 4]],
            norm_dir=0,
            elevation=0.0,
            length=0.035,
        )

        content = export(csx, tmp_path)

        assert "%LNside*%" in content
        assert "G36*" not in content
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
