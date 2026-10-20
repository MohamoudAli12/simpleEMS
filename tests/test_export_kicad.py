"""Tests for :func:`simpleEMS.export_layout.export_kicad_footprint`.

A footprint is an s-expression, so the tests parse it into nested lists and
check what KiCad would place: the origin at the feed, the copper split into
the feed pad and the ground pad, the vias as plated holes, and Y mirrored to
KiCad's downward axis. Layer inference itself is pinned in
``test_export_gerber.py``.
"""

import re
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from CSXCAD import ContinuousStructure  # noqa: E402

from simpleEMS.export_layout import export_kicad_footprint  # noqa: E402
from simpleEMS.sim_tools import SimTools  # noqa: E402

THICKNESS = 1.6
COPPER = 0.035
# A lumped port across the substrate at the end of a 1 mm wide feed at y = -4
FEED_PORT = SimpleNamespace(
    start=[0.5, -4.0, -COPPER], stop=[-0.5, -4.0, THICKNESS + COPPER]
)


@pytest.fixture
def board():
    """A patch with a feed on top, ground on the bottom with an antipad, a
    shorting via, and the substrate."""
    csx = ContinuousStructure()

    substrate = csx.AddMaterial("substrate", epsilon=4.4)
    substrate.AddBox(priority=0, start=[-5, -5, 0], stop=[5, 5, THICKNESS])

    patch = csx.AddMetal("patch")
    patch.AddBox(priority=1, start=[-2, 0, THICKNESS], stop=[2, 3, THICKNESS + COPPER])

    feed = csx.AddMetal("feed")
    feed.AddBox(
        priority=1, start=[-0.5, -4, THICKNESS], stop=[0.5, 0, THICKNESS + COPPER]
    )

    ground = csx.AddMetal("ground")
    ground.AddBox(priority=1, start=[-5, -5, -COPPER], stop=[5, 5, 0])

    via = csx.AddMetal("via")
    via.AddCylinder(
        priority=2, start=[2, 2, -COPPER], stop=[2, 2, THICKNESS + COPPER], radius=0.2
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


def parse(text: str) -> list:
    """Parse an s-expression into nested lists of strings."""
    tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', text)
    stack: list[list] = [[]]
    for token in tokens:
        if token == "(":
            stack.append([])
        elif token == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token.strip('"'))
    (root,) = stack[0]
    return root


def children(node: list, name: str) -> list[list]:
    return [child for child in node if isinstance(child, list) and child[0] == name]


def child(node: list, name: str) -> list:
    (found,) = children(node, name)
    return found


def numbers(node: list) -> list[float]:
    return [float(value) for value in node[1:]]


def polygons(pad: list) -> list[list[tuple[float, float]]]:
    """Each gr_poly of a custom pad, in footprint coordinates."""
    pad_x, pad_y = numbers(child(pad, "at"))
    return [
        [
            (float(point[1]) + pad_x, float(point[2]) + pad_y)
            for point in child(poly, "pts")[1:]
        ]
        for poly in children(child(pad, "primitives"), "gr_poly")
    ]


@pytest.fixture
def footprint(board, tmp_path):
    path = export_kicad_footprint(board, FEED_PORT, tmp_path, {}, name="patch")
    return parse(path.read_text())


def pads(footprint, number, layer=None):
    return [
        pad
        for pad in children(footprint, "pad")
        if pad[1] == number and (layer is None or child(pad, "layers")[1] == layer)
    ]


class TestFile:
    def test_writes_name_kicad_mod(self, board, tmp_path):
        path = export_kicad_footprint(board, FEED_PORT, tmp_path, {}, name="patch")

        assert path == tmp_path / "patch.kicad_mod"

    def test_parses_as_one_footprint(self, footprint):
        assert footprint[:2] == ["footprint", "patch"]

    def test_parentheses_balance(self, board, tmp_path):
        text = export_kicad_footprint(board, FEED_PORT, tmp_path, {}).read_text()

        assert text.count("(") == text.count(")")


class TestFeedPad:
    def test_sits_at_the_origin_on_the_top_layer(self, footprint):
        (feed,) = pads(footprint, "1", "F.Cu")

        assert numbers(child(feed, "at")) == [0, 0]

    def test_is_a_square_the_width_of_the_port(self, footprint):
        (feed,) = pads(footprint, "1", "F.Cu")

        assert numbers(child(feed, "size")) == [1.0, 1.0]

    def test_holds_the_patch_and_the_feed(self, footprint):
        (feed,) = pads(footprint, "1", "F.Cu")

        assert len(polygons(feed)) == 2

    def test_feed_trace_ends_at_the_origin_and_runs_up_the_page(self, footprint):
        (feed,) = pads(footprint, "1", "F.Cu")
        trace = polygons(feed)[1]

        # Simulation y from -4 (the port) to 0 is footprint y from 0 to -4.
        assert sorted({y for _, y in trace}) == [-4.0, 0.0]
        assert sorted({x for x, _ in trace}) == [-0.5, 0.5]


class TestGroundPad:
    def test_ground_copper_is_pad_two_on_the_bottom(self, footprint):
        (ground,) = pads(footprint, "2", "B.Cu")
        (outline,) = polygons(ground)

        assert sorted(set(outline)) == [(-5, -9), (-5, 1), (5, -9), (5, 1)]

    def test_via_is_a_plated_hole_on_the_ground_pad(self, footprint):
        (via,) = [pad for pad in pads(footprint, "2") if pad[2] == "thru_hole"]

        assert numbers(child(via, "at")) == [2.0, -6.0]
        assert numbers(child(via, "drill")) == [0.4]
        assert numbers(child(via, "size")) == pytest.approx([0.7, 0.7])

    def test_custom_ground_names(self, board, tmp_path):
        path = export_kicad_footprint(board, FEED_PORT, tmp_path, {}, ground=["patch"])
        footprint = parse(path.read_text())

        # The patch moves to pad 2; the old ground joins the feed net.
        assert len(pads(footprint, "2", "F.Cu")) == 1
        assert len(pads(footprint, "1", "B.Cu")) == 1


class TestOutline:
    def test_fab_and_courtyard_cover_the_substrate(self, footprint):
        rectangles = children(footprint, "fp_rect")

        assert [child(rect, "layer")[1] for rect in rectangles] == ["F.Fab", "F.CrtYd"]
        for rect in rectangles:
            assert numbers(child(rect, "start")) == [-5, -9]
            assert numbers(child(rect, "end")) == [5, 1]


class TestOptions:
    def test_ignored_properties_are_left_out(self, board, tmp_path):
        path = export_kicad_footprint(
            board, FEED_PORT, tmp_path, {"ignore": ["ground"]}
        )

        assert pads(parse(path.read_text()), "2", "B.Cu") == []

    def test_dropped_antipads_are_named_in_a_warning(self, board, tmp_path, capsys):
        export_kicad_footprint(board, FEED_PORT, tmp_path, {})

        assert "antipad" in capsys.readouterr().out


class TestSimToolsWrapper:
    def test_writes_into_a_kicad_footprint_folder(self, built_inset, tmp_path):
        _antenna, sim, _params, port = built_inset

        path = SimTools.export_kicad_footprint(sim, port, output_path=tmp_path)

        assert path == tmp_path / "kicad_footprint" / "antenna.kicad_mod"

    def test_inset_patch_feeds_on_top_and_grounds_on_the_bottom(
        self, built_inset, tmp_path
    ):
        _antenna, sim, _params, port = built_inset

        path = SimTools.export_kicad_footprint(sim, port, output_path=tmp_path)
        footprint = parse(path.read_text())

        assert len(pads(footprint, "1", "F.Cu")) == 1
        assert len(pads(footprint, "2", "B.Cu")) == 1
