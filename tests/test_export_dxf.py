"""Tests for :func:`simpleEMS.export_layout.export_dxf`.

DXF is plain text made of (group code, value) line pairs, so the tests parse
it back into entities and check what a CAD tool would import: which layer
each shape lands on and where its vertices are. Layer inference itself is
pinned in ``test_export_gerber.py``; this file covers the DXF writing only.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from CSXCAD import ContinuousStructure  # noqa: E402

from simpleEMS.export_layout import export_dxf  # noqa: E402
from simpleEMS.sim_tools import SimTools  # noqa: E402

THICKNESS = 1.6
COPPER = 0.035


@pytest.fixture
def board():
    """A two-layer board: a top pad, an L-shaped polygon, bottom ground with
    an antipad, one through via, and the substrate."""
    csx = ContinuousStructure()

    substrate = csx.AddMaterial("substrate", epsilon=4.4)
    substrate.AddBox(priority=0, start=[-5, -5, 0], stop=[5, 5, THICKNESS])

    pad = csx.AddMetal("pad")
    pad.AddBox(
        priority=1,
        start=[-1, -2, THICKNESS],
        stop=[1, 2, THICKNESS + COPPER],
    )

    trace = csx.AddMetal("trace")
    trace.AddLinPoly(
        priority=1,
        points=[[0, 3, 3, 1, 1, 0], [3, 3, 4, 4, 5, 5]],
        norm_dir=2,
        elevation=THICKNESS,
        length=COPPER,
    )

    ground = csx.AddMetal("ground")
    ground.AddBox(priority=1, start=[-5, -5, -COPPER], stop=[5, 5, 0])

    via = csx.AddMetal("via")
    via.AddCylinder(
        priority=2, start=[2, 2, -COPPER], stop=[2, 2, THICKNESS + COPPER], radius=0.25
    )

    antipad = csx.AddMaterial("antipad", epsilon=1)
    antipad.AddLinPoly(
        priority=3,
        points=[[-3, -2, -2, -3], [-3, -3, -2, -2]],
        norm_dir=2,
        elevation=-COPPER,
        length=COPPER,
    )
    return csx


def read_entities(path: Path) -> list[dict]:
    """Parse the ENTITIES section into dicts: type, layer, vertices, circle."""
    lines = path.read_text().split("\n")
    pairs = list(zip(lines[0::2], lines[1::2], strict=False))
    start = pairs.index(("2", "ENTITIES"))
    entities: list[dict] = []
    for code, value in pairs[start + 1 :]:
        if code == "0":
            if value in ("VERTEX", "SEQEND"):
                entity = {"type": value}
                if value == "VERTEX":
                    entities[-1].setdefault("vertices", []).append(entity)
                continue
            entity = {"type": value}
            entities.append(entity)
        elif code == "8" and entity["type"] not in ("VERTEX", "SEQEND"):
            entity["layer"] = value
        elif code in ("10", "20", "40"):
            entity[code] = float(value)
    for entity in entities:
        if "vertices" in entity:
            entity["points"] = [
                (vertex["10"], vertex["20"]) for vertex in entity["vertices"]
            ]
    return [entity for entity in entities if entity["type"] not in ("ENDSEC", "EOF")]


class TestFile:
    def test_writes_layout_dxf(self, board, tmp_path):
        path = export_dxf(board, tmp_path, {})

        assert path == tmp_path / "layout.dxf"

    def test_custom_filename(self, board, tmp_path):
        path = export_dxf(board, tmp_path, {}, filename="patch")

        assert path == tmp_path / "patch.dxf"

    def test_is_r12_in_millimetres(self, board, tmp_path):
        text = export_dxf(board, tmp_path, {}).read_text()

        assert "$ACADVER\n1\nAC1009\n" in text
        assert "$INSUNITS\n70\n4\n" in text

    def test_ends_with_eof(self, board, tmp_path):
        assert export_dxf(board, tmp_path, {}).read_text().endswith("0\nEOF\n")

    def test_every_used_layer_is_declared(self, board, tmp_path):
        path = export_dxf(board, tmp_path, {})
        text = path.read_text()

        for layer in {entity["layer"] for entity in read_entities(path)}:
            assert f"0\nLAYER\n2\n{layer}\n" in text


class TestEntities:
    @pytest.fixture
    def entities(self, board, tmp_path):
        return read_entities(export_dxf(board, tmp_path, {}))

    def by_layer(self, entities, layer):
        return [entity for entity in entities if entity.get("layer") == layer]

    def test_top_copper_is_two_closed_polylines(self, entities):
        top = self.by_layer(entities, "F_Cu")

        assert [entity["type"] for entity in top] == ["POLYLINE", "POLYLINE"]

    def test_box_corners(self, entities):
        pad = self.by_layer(entities, "F_Cu")[0]

        assert pad["points"] == [(-1, -2), (1, -2), (1, 2), (-1, 2)]

    def test_polygon_keeps_every_vertex(self, entities):
        trace = self.by_layer(entities, "F_Cu")[1]

        assert trace["points"] == [(0, 3), (3, 3), (3, 4), (1, 4), (1, 5), (0, 5)]

    def test_ground_is_on_the_bottom_layer(self, entities):
        (ground,) = self.by_layer(entities, "B_Cu")

        assert ground["points"] == [(-5, -5), (5, -5), (5, 5), (-5, 5)]

    def test_antipad_gets_its_own_layer(self, entities):
        (antipad,) = self.by_layer(entities, "B_Cu_Clearance")

        assert antipad["points"] == [(-3, -3), (-2, -3), (-2, -2), (-3, -2)]

    def test_outline_is_the_substrate(self, entities):
        (outline,) = self.by_layer(entities, "Edge_Cuts")

        assert outline["points"] == [(-5, -5), (5, -5), (5, 5), (-5, 5)]

    def test_via_is_a_circle_at_the_conductor_diameter(self, entities):
        (via,) = self.by_layer(entities, "Drills")

        assert via["type"] == "CIRCLE"
        assert (via["10"], via["20"], via["40"]) == (2, 2, 0.25)

    def test_ignored_properties_are_left_out(self, board, tmp_path):
        entities = read_entities(export_dxf(board, tmp_path, {"ignore": ["ground"]}))

        assert self.by_layer(entities, "B_Cu") == []


class TestSimToolsWrapper:
    def test_writes_into_a_dxf_subdirectory(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset

        path = SimTools.export_dxf(sim, output_path=tmp_path)

        assert path == tmp_path / "dxf" / "layout.dxf"

    def test_inset_patch_has_top_and_bottom_copper(self, built_inset, tmp_path):
        _antenna, sim, _params, _port = built_inset

        entities = read_entities(SimTools.export_dxf(sim, output_path=tmp_path))

        assert {entity["layer"] for entity in entities} == {
            "F_Cu",
            "B_Cu",
            "Edge_Cuts",
        }
