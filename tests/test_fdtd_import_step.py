"""Tests for the STEP-to-FDTD import path.

``fdtd_import_step`` rebuilds the named solids of a STEP file as CSXCAD
geometry so a CAD model can be simulated without hand-building it. The module
is a prototype and most of its risk is in the reconstruction rather than the
simulation: a solid that should have come back as a native box arriving as a
tessellated polyhedron, or a port whose STEP-mandated minimum thickness is
mistaken for real geometry, both produce a structure that meshes and solves
and quietly models the wrong thing.

Those two rules -- box detection and port flattening -- carry long comments in
the source explaining why they exist, so they are pinned here in detail. The
solve itself is covered by one ``slow`` test at the bottom.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.needs_csxcad, pytest.mark.needs_cadquery]

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the markers above keep it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")
pytest.importorskip("cadquery")

import cadquery as cq  # noqa: E402

from simpleEMS.fdtd_import_step import (  # noqa: E402
    _DEGENERATE_PORT_THICKNESS_MM,
    _axis_aligned_box,
    _combined_bbox_mm,
    _load_named_solids,
    _port_box,
    _quantize_f32,
    _tessellate_to_stl,
    simulate_step_FDTD,
    StepFDTDParams,
)


def box_at(x, y, z, w, h, d):
    """A box of the given size with its minimum corner at ``(x, y, z)``."""
    return cq.Solid.makeBox(w, h, d).locate(cq.Location(cq.Vector(x, y, z)))


@pytest.fixture
def microstrip_step(tmp_path):
    """A STEP file holding a grounded microstrip line and a port marker.

    Named solids, as :func:`~simpleEMS.export_cad.export_step` would write
    them. ``p1`` is the thin marker solid that stands in for a lumped port:
    0.001 mm in y, which is the minimum thickness STEP can represent.
    """
    asm = cq.Assembly(name="model")
    asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
    asm.add(box_at(-10, -10, -1.635, 20, 20, 0.035), name="ground")
    asm.add(box_at(-1.5, -8, 0, 3, 16, 0.035), name="trace")
    asm.add(box_at(-1.5, -8, -1.6, 3, 0.001, 1.6), name="p1")
    path = tmp_path / "microstrip.step"
    asm.save(str(path))
    return path


@pytest.fixture
def build_only(monkeypatch):
    """Run ``simulate_step_FDTD`` up to, but not including, post-processing.

    The geometry reconstruction is what this module is responsible for; the
    post-processing belongs to ``sim_tools`` and needs solver output that a
    build-only test has no reason to produce.
    """
    captured = {}

    def fake_compute(sim, port, output_path=None):
        captured["sim"] = sim
        captured["port"] = port
        captured["output_path"] = output_path
        return "sim-data-sentinel"

    monkeypatch.setattr(
        "simpleEMS.fdtd_import_step.SimTools.compute_sim_data", fake_compute
    )
    return captured


# ---------------------------------------------------------------------
# _quantize_f32
# ---------------------------------------------------------------------
class TestQuantizeF32:
    def test_exact_binary_values_are_unchanged(self):
        for value in (0.0, 0.5, 1.0, -2.0, 0.25):
            assert _quantize_f32(value) == value

    def test_rounds_to_single_precision(self):
        """1.6 is not representable in binary32; the nearest float is slightly
        above it."""
        assert _quantize_f32(1.6) != 1.6
        assert _quantize_f32(1.6) == pytest.approx(1.6, abs=1e-7)

    def test_is_idempotent(self):
        """Applied twice it must not drift, or a coordinate quantised on one
        path would not match the same coordinate quantised on another."""
        for value in (1.6, 0.035, -3.14159, 1e-3):
            once = _quantize_f32(value)
            assert _quantize_f32(once) == once

    def test_matches_what_a_tessellated_edge_lands_on(self):
        """The whole point: an STL vertex is a binary32, so a hand-built
        coordinate has to be rounded the same way to compare equal."""
        value = 1.6
        as_stl = np.float32(value).astype(np.float64)

        assert _quantize_f32(value) == as_stl

    def test_preserves_sign(self):
        assert _quantize_f32(-1.6) == -_quantize_f32(1.6)

    def test_handles_zero_and_tiny_values(self):
        assert _quantize_f32(0.0) == 0.0
        assert _quantize_f32(1e-3) == pytest.approx(1e-3, abs=1e-10)


# ---------------------------------------------------------------------
# _axis_aligned_box
# ---------------------------------------------------------------------
class TestAxisAlignedBox:
    def test_a_box_is_recognised(self):
        start, stop = _axis_aligned_box(box_at(-1, -2, 0, 2, 4, 0.035))

        assert start == pytest.approx([-1.0, -2.0, 0.0])
        assert stop == pytest.approx([1.0, 2.0, 0.035], abs=1e-7)

    def test_corners_are_quantised(self):
        """So the box's faces land on the same values a tessellated neighbour's
        vertices do, and the mesher sees one boundary rather than two."""
        _start, stop = _axis_aligned_box(box_at(0, 0, 0, 1.6, 1, 1))

        assert stop[0] == _quantize_f32(1.6)

    def test_a_cylinder_is_not_a_box(self):
        assert _axis_aligned_box(cq.Solid.makeCylinder(1.0, 2.0)) is None

    def test_a_sphere_is_not_a_box(self):
        assert _axis_aligned_box(cq.Solid.makeSphere(1.0)) is None

    def test_a_rotated_box_is_not_a_box(self):
        """It has six faces but does not fill its own extents, and rebuilding
        it as an axis-aligned box would silently straighten it."""
        rotated = cq.Solid.makeBox(2, 4, 1).rotate(
            cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 30
        )

        assert _axis_aligned_box(rotated) is None

    def test_a_solid_with_a_notch_is_not_a_box(self):
        """This is the case the source comment is about: importing a notched
        metal as a plain box loses the notch, and with it the mesh refinement
        at its edge."""
        notched = (
            cq.Workplane("XY")
            .box(10, 10, 1)
            .faces(">Z")
            .workplane()
            .rect(2, 2)
            .cutBlind(-0.5)
            .val()
        )

        assert _axis_aligned_box(notched) is None

    def test_a_flat_sheet_is_still_a_box(self):
        """Zero-thickness sheets come back from STEP with a 1e-3 mm floor."""
        result = _axis_aligned_box(box_at(0, 0, 0, 10, 10, 1e-3))

        assert result is not None

    def test_a_degenerate_solid_is_rejected(self):
        """Volume zero means no box to measure."""

        class Degenerate:
            def Faces(self):  # noqa: N802 - mirrors the cadquery API
                return [None] * 6

            def BoundingBox(self):  # noqa: N802
                class BB:
                    xmin = xmax = ymin = ymax = zmin = zmax = 0.0

                return BB()

            def Volume(self):  # noqa: N802
                return 0.0

        assert _axis_aligned_box(Degenerate()) is None

    def test_an_unreadable_solid_is_rejected_rather_than_raising(self):
        """A malformed import should drop to the tessellated path, not abort
        the whole build."""

        class Broken:
            def Faces(self):  # noqa: N802
                raise ValueError("bad shape")

        assert _axis_aligned_box(Broken()) is None

    def test_survives_a_step_round_trip(self, microstrip_step):
        solids = _load_named_solids(microstrip_step)

        for name in ("substrate", "ground", "trace", "p1"):
            assert _axis_aligned_box(solids[name]) is not None


# ---------------------------------------------------------------------
# _port_box
# ---------------------------------------------------------------------
class TestPortBox:
    def bb(self, x, y, z, w, h, d):
        return box_at(x, y, z, w, h, d).BoundingBox()

    def test_a_thin_axis_is_collapsed_to_a_plane(self):
        """The 0.001 mm is STEP's minimum-thickness artefact, not geometry;
        left alone the mesher over-refines it like a real feature."""
        start, stop = _port_box(self.bb(0, 0, 0, 3, 0.001, 1.6), "z")

        assert start[1] == stop[1]

    def test_the_collapsed_plane_sits_at_the_midpoint(self):
        start, stop = _port_box(self.bb(0, 5, 0, 3, 0.001, 1.6), "z")

        assert start[1] == pytest.approx(5.0005, abs=1e-6)
        assert stop[1] == start[1]

    def test_the_excitation_axis_is_never_collapsed(self):
        """Collapsing it would leave the port no gap to drive across."""
        start, stop = _port_box(self.bb(0, 0, 0, 3, 1.6, 0.001), "z")

        assert stop[2] > start[2]

    def test_a_genuinely_wide_axis_is_kept(self):
        """The trace width is real geometry and has to stay."""
        start, stop = _port_box(self.bb(0, 0, 0, 3, 0.001, 1.6), "z")

        assert stop[0] - start[0] == pytest.approx(3.0)

    def test_the_threshold_sits_between_the_artefact_and_real_geometry(self):
        """1e-3 mm is the STEP floor; 0.1 mm is the narrowest real trace this
        project builds. The cut has to separate them."""
        assert 1e-3 < _DEGENERATE_PORT_THICKNESS_MM < 0.1

    def test_an_axis_just_under_the_threshold_collapses(self):
        thin = _DEGENERATE_PORT_THICKNESS_MM * 0.5
        start, stop = _port_box(self.bb(0, 0, 0, 3, thin, 1.6), "z")

        assert start[1] == stop[1]

    def test_an_axis_just_over_the_threshold_survives(self):
        thick = _DEGENERATE_PORT_THICKNESS_MM * 2
        start, stop = _port_box(self.bb(0, 0, 0, 3, thick, 1.6), "z")

        assert stop[1] > start[1]

    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    def test_every_direction_is_accepted(self, direction):
        start, stop = _port_box(self.bb(0, 0, 0, 3, 0.001, 1.6), direction)

        assert len(start) == 3 and len(stop) == 3

    def test_corners_are_quantised(self):
        start, stop = _port_box(self.bb(0, 0, 0, 1.6, 1.6, 1.6), "z")

        assert stop[0] == _quantize_f32(1.6)

    def test_corners_are_ordered_low_to_high(self):
        start, stop = _port_box(self.bb(-3, -2, -1, 6, 4, 2), "z")

        assert all(a <= b for a, b in zip(start, stop, strict=True))


# ---------------------------------------------------------------------
# StepFDTDParams
# ---------------------------------------------------------------------
class TestStepFDTDParams:
    @pytest.fixture
    def params(self):
        return StepFDTDParams(
            freqs=np.linspace(2e9, 3e9, 11),
            struct_bbox_mm=(-10.0, 10.0, -8.0, 8.0, -1.6, 0.035),
            substrate_eps_r=4.4,
        )

    def test_freq_range_is_the_span_of_the_points(self, params):
        assert params.freq_range == (2e9, 3e9)

    def test_freq_range_does_not_assume_sorted_input(self, params):
        unsorted = StepFDTDParams(
            freqs=np.array([3e9, 1e9, 2e9]),
            struct_bbox_mm=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        )

        assert unsorted.freq_range == (1e9, 3e9)

    def test_main_freq_is_the_centre_of_the_range(self, params):
        assert params.main_freq == pytest.approx(2.5e9)

    def test_width_and_length_come_from_the_bounding_box(self, params):
        assert params.substrate_width_mm == pytest.approx(20.0)
        assert params.substrate_length_mm == pytest.approx(16.0)

    def test_simulation_box_has_three_dimensions(self, params):
        assert np.asarray(params.simulation_box).shape[-1] == 3

    def test_simulation_box_encloses_the_structure(self, params):
        box = np.asarray(params.simulation_box)

        assert box.max() >= params.substrate_width_mm / 2

    def test_permittivity_drives_the_wavelength(self):
        """Left at the air default, lambda0 would be the vacuum wavelength --
        overpadding the box and under-resolving the metal edges."""
        loaded = StepFDTDParams(
            freqs=np.linspace(2e9, 3e9, 11),
            struct_bbox_mm=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            substrate_eps_r=4.4,
        )
        air = StepFDTDParams(
            freqs=np.linspace(2e9, 3e9, 11),
            struct_bbox_mm=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        )

        assert loaded.lambda0 < air.lambda0

    def test_defaults_describe_vacuum(self):
        params = StepFDTDParams(
            freqs=np.linspace(2e9, 3e9, 11),
            struct_bbox_mm=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        )

        assert params.substrate_eps_r == 1.0
        assert params.substrate_tand == 0.0
        assert params.substrate_thickness_mm == 0.0

    def test_mesh_resolution_follows_the_wavelength(self, params):
        assert params.FDTD_mesh_resolution == pytest.approx(
            params.lambda0 / params.FDTD_mesh_resolution_factor
        )

    def test_is_a_simparams(self, params):
        from simpleEMS.sim_params import SimParams

        assert isinstance(params, SimParams)

    def test_is_keyword_only(self):
        with pytest.raises(TypeError):
            StepFDTDParams(np.linspace(2e9, 3e9, 11), (0, 1, 0, 1, 0, 1))


# ---------------------------------------------------------------------
# STEP reading helpers
# ---------------------------------------------------------------------
class TestStepHelpers:
    def test_solids_come_back_keyed_by_name(self, microstrip_step):
        solids = _load_named_solids(microstrip_step)

        assert set(solids) == {"substrate", "ground", "trace", "p1"}

    def test_solids_are_cadquery_solids(self, microstrip_step):
        """``_axis_aligned_box`` calls Faces/Volume/BoundingBox on these."""
        solids = _load_named_solids(microstrip_step)

        for solid in solids.values():
            assert hasattr(solid, "Faces")
            assert hasattr(solid, "Volume")

    def test_accepts_a_string_path(self, microstrip_step):
        assert _load_named_solids(str(microstrip_step))

    def test_combined_bbox_spans_every_solid(self, microstrip_step):
        solids = _load_named_solids(microstrip_step)

        xmin, xmax, ymin, ymax, zmin, zmax = _combined_bbox_mm(list(solids.values()))

        assert (xmin, xmax) == pytest.approx((-10.0, 10.0))
        assert (ymin, ymax) == pytest.approx((-10.0, 10.0))
        assert zmin == pytest.approx(-1.635, abs=1e-6)
        assert zmax == pytest.approx(0.035, abs=1e-6)

    def test_combined_bbox_of_one_solid_is_its_own(self):
        solid = box_at(1, 2, 3, 4, 5, 6)

        assert _combined_bbox_mm([solid]) == pytest.approx((1, 5, 2, 7, 3, 9))

    def test_tessellation_writes_an_stl(self, tmp_path):
        path = _tessellate_to_stl(cq.Solid.makeCylinder(1.0, 2.0), tmp_path, "post")

        assert path == tmp_path / "post.stl"
        assert path.stat().st_size > 0

    def test_the_stl_is_well_formed(self, tmp_path):
        """Binary STL is an 84-byte header plus 50 bytes per facet. CSXCAD's
        PolyhedronReader rejects a file whose declared count and length
        disagree, and the rejection surfaces much later as empty geometry."""
        path = _tessellate_to_stl(cq.Solid.makeCylinder(1.0, 2.0), tmp_path, "post")
        content = path.read_bytes()

        declared = int.from_bytes(content[80:84], "little")

        assert declared > 0
        assert len(content) == 84 + 50 * declared

    def test_a_finer_tolerance_gives_more_triangles(self, tmp_path):
        coarse = _tessellate_to_stl(
            cq.Solid.makeCylinder(1.0, 2.0), tmp_path, "coarse", tolerance=1e-1
        )
        fine = _tessellate_to_stl(
            cq.Solid.makeCylinder(1.0, 2.0), tmp_path, "fine", tolerance=1e-4
        )

        assert fine.stat().st_size > coarse.stat().st_size


# ---------------------------------------------------------------------
# simulate_step_FDTD -- argument handling and reconstruction
# ---------------------------------------------------------------------
class TestSimulateStepFDTD:
    def run_build(self, step, tmp_path, **overrides):
        kwargs = dict(
            step_file=step,
            freqs=np.linspace(2e9, 3e9, 11),
            dielectrics={"substrate": (4.4, 0.001)},
            pec=["ground", "trace"],
            ports={"p1": {"direction": "z"}},
            output_path=tmp_path / "out",
            num_points=11,
            show_structure=False,
            run=False,
        )
        kwargs.update(overrides)
        return simulate_step_FDTD(**kwargs)

    def test_portless_call_is_rejected(self, microstrip_step, tmp_path):
        """Without a port there is nothing to excite or measure."""
        with pytest.raises(RuntimeError, match="at least one entry in `ports`"):
            self.run_build(microstrip_step, tmp_path, ports={})

    def test_an_unknown_solid_name_is_rejected(self, microstrip_step, tmp_path):
        with pytest.raises(KeyError, match="nonexistent"):
            self.run_build(microstrip_step, tmp_path, pec=["ground", "nonexistent"])

    def test_the_error_names_the_step_file(self, microstrip_step, tmp_path):
        with pytest.raises(KeyError, match="microstrip.step"):
            self.run_build(microstrip_step, tmp_path, pec=["nope"])

    def test_an_unknown_dielectric_is_rejected(self, microstrip_step, tmp_path):
        with pytest.raises(KeyError):
            self.run_build(
                microstrip_step, tmp_path, dielectrics={"missing": (4.4, 0.001)}
            )

    def test_an_unknown_port_is_rejected(self, microstrip_step, tmp_path):
        with pytest.raises(KeyError):
            self.run_build(microstrip_step, tmp_path, ports={"p9": {}})

    def test_run_false_still_post_processes(self, microstrip_step, tmp_path):
        """``run=False`` skips the solver but not the post-processing, so
        against an empty directory it fails on the missing port probes rather
        than returning a built-but-unsolved structure. Pinned because the
        parameter reads like a build-only switch."""
        with pytest.raises(FileNotFoundError, match="port_ut_1"):
            self.run_build(microstrip_step, tmp_path)

    def test_every_named_solid_becomes_a_property(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path)

        csx = build_only["sim"].CSX
        names = [csx.GetProperty(i).GetName() for i in range(csx.GetQtyProperties())]
        assert {"substrate", "ground", "trace"} <= set(names)

    def test_unnamed_solids_are_left_out(self, microstrip_step, tmp_path, build_only):
        """Anything not listed in dielectrics/pec/ports is not simulated."""
        self.run_build(microstrip_step, tmp_path, pec=["ground"])

        csx = build_only["sim"].CSX
        names = [csx.GetProperty(i).GetName() for i in range(csx.GetQtyProperties())]
        assert "trace" not in names

    def test_the_port_becomes_a_lumped_element(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path)

        csx = build_only["sim"].CSX
        names = [csx.GetProperty(i).GetName() for i in range(csx.GetQtyProperties())]
        assert "port_resist_1" in names

    def test_the_port_solid_is_not_also_added_as_geometry(
        self, microstrip_step, tmp_path, build_only
    ):
        """It only marks where the port goes; rebuilding it would put a stray
        body in the model."""
        self.run_build(microstrip_step, tmp_path)

        csx = build_only["sim"].CSX
        names = [csx.GetProperty(i).GetName() for i in range(csx.GetQtyProperties())]
        assert "p1" not in names

    def test_a_single_port_is_passed_on_its_own(
        self, microstrip_step, tmp_path, build_only
    ):
        """``compute_sim_data`` takes a bare port for one-port runs and a list
        for two-port runs, and branches on the type."""
        self.run_build(microstrip_step, tmp_path)

        assert not isinstance(build_only["port"], list)

    def test_the_dielectric_gets_the_requested_permittivity(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path)

        csx = build_only["sim"].CSX
        substrate = next(
            csx.GetProperty(i)
            for i in range(csx.GetQtyProperties())
            if csx.GetProperty(i).GetName() == "substrate"
        )
        assert substrate.GetMaterialProperty("epsilon") == pytest.approx(4.4)

    def test_a_lossless_dielectric_gets_zero_conductivity(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path, dielectrics={"substrate": (4.4, 0.0)})

        csx = build_only["sim"].CSX
        substrate = next(
            csx.GetProperty(i)
            for i in range(csx.GetQtyProperties())
            if csx.GetProperty(i).GetName() == "substrate"
        )
        assert substrate.GetMaterialProperty("kappa") == pytest.approx(0.0)

    def test_the_mesh_is_generated(self, microstrip_step, tmp_path, build_only):
        self.run_build(microstrip_step, tmp_path)

        grid = build_only["sim"].CSX.GetGrid()
        for dim in range(3):
            assert len(grid.GetLines(dim)) > 1

    def test_the_output_directory_is_created(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path)

        assert (tmp_path / "out").is_dir()

    def test_the_output_path_reaches_post_processing(
        self, microstrip_step, tmp_path, build_only
    ):
        self.run_build(microstrip_step, tmp_path)

        assert build_only["output_path"] == tmp_path / "out"

    def test_returns_the_data_and_the_far_field_object(
        self, microstrip_step, tmp_path, build_only
    ):
        from openEMS.nf2ff import nf2ff

        data, far_field = self.run_build(microstrip_step, tmp_path)

        assert data == "sim-data-sentinel"
        assert isinstance(far_field, nf2ff)

    def line_count(self, build_only):
        grid = build_only["sim"].CSX.GetGrid()
        return sum(len(grid.GetLines(dim)) for dim in range(3))

    @pytest.fixture
    def two_dielectric_step(self, tmp_path):
        """Same geometry as ``microstrip_step`` plus a second dielectric block,
        so the permittivity set can be varied without moving any solid."""
        asm = cq.Assembly(name="model")
        asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
        asm.add(box_at(-10, -10, 0.035, 20, 20, 0.4), name="spacer")
        asm.add(box_at(-10, -10, -1.635, 20, 20, 0.035), name="ground")
        asm.add(box_at(-1.5, -8, 0, 3, 16, 0.035), name="trace")
        asm.add(box_at(-1.5, -8, -1.6, 3, 0.001, 1.6), name="p1")
        path = tmp_path / "two_dielectric.step"
        asm.save(str(path))
        return path

    def test_the_permittivity_used_is_the_most_loaded_one(
        self, two_dielectric_step, tmp_path, build_only
    ):
        """``substrate_eps_r`` sets lambda0 and with it the mesh density, so it
        has to be the highest imported value -- not the first, the last, or the
        mean. The geometry is identical across these three builds, so any
        difference in the grid comes from the permittivity rule alone.
        """
        counts = {}
        for label, eps in {
            "max_is_first": {"substrate": (4.4, 0.001), "spacer": (2.0, 0.001)},
            "max_is_last": {"substrate": (2.0, 0.001), "spacer": (4.4, 0.001)},
            "both_at_max": {"substrate": (4.4, 0.001), "spacer": (4.4, 0.001)},
        }.items():
            self.run_build(two_dielectric_step, tmp_path, dielectrics=eps)
            counts[label] = self.line_count(build_only)

        assert counts["max_is_first"] == counts["both_at_max"]
        assert counts["max_is_last"] == counts["both_at_max"]

    def test_a_more_loaded_dielectric_refines_the_mesh(
        self, microstrip_step, tmp_path, build_only
    ):
        """The consequence of the rule above: a shorter wavelength in the
        substrate has to produce a finer grid, or the metal edges are
        under-resolved."""
        self.run_build(
            microstrip_step, tmp_path, dielectrics={"substrate": (2.0, 0.001)}
        )
        light = self.line_count(build_only)

        self.run_build(
            microstrip_step, tmp_path, dielectrics={"substrate": (10.0, 0.001)}
        )
        heavy = self.line_count(build_only)

        assert heavy > light

    def test_a_tessellated_solid_is_written_out(self, tmp_path, build_only):
        """Non-box solids go through STL because CSXCAD cannot read STEP."""
        asm = cq.Assembly(name="model")
        asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
        asm.add(cq.Solid.makeCylinder(2.0, 0.035), name="disc")
        asm.add(box_at(-1.5, -8, -1.6, 3, 0.001, 1.6), name="p1")
        step = tmp_path / "curved.step"
        asm.save(str(step))

        simulate_step_FDTD(
            step_file=step,
            freqs=np.linspace(2e9, 3e9, 11),
            dielectrics={"substrate": (4.4, 0.001)},
            pec=["disc"],
            ports={"p1": {"direction": "z"}},
            output_path=tmp_path / "out",
            num_points=11,
            show_structure=False,
            run=False,
        )

        assert (tmp_path / "out" / "step_stl" / "disc.stl").is_file()

    def test_show_structure_is_honoured(
        self, microstrip_step, tmp_path, build_only, monkeypatch
    ):
        """The viewer blocks on a GUI window, so it must not fire by default in
        a headless run."""
        shown = []
        monkeypatch.setattr(
            "simpleEMS.fdtd_import_step.SimTools.write_and_show_structure",
            lambda sim, path: shown.append(path),
        )

        self.run_build(microstrip_step, tmp_path, show_structure=True)

        assert shown == [tmp_path / "out"]

    def test_ports_are_ordered_by_number_not_by_dict_order(self, tmp_path, build_only):
        """``compute_sim_data`` reads ``port[0]`` as the driven port, so the
        list has to be sorted by port number however the dict was written."""
        asm = cq.Assembly(name="model")
        asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
        asm.add(box_at(-10, -10, -1.635, 20, 20, 0.035), name="ground")
        asm.add(box_at(-1.5, -8, 0, 3, 16, 0.035), name="trace")
        asm.add(box_at(-1.5, -8, -1.6, 3, 0.001, 1.6), name="pa")
        asm.add(box_at(-1.5, 8, -1.6, 3, 0.001, 1.6), name="pb")
        step = tmp_path / "twoport.step"
        asm.save(str(step))

        simulate_step_FDTD(
            step_file=step,
            freqs=np.linspace(2e9, 3e9, 11),
            dielectrics={"substrate": (4.4, 0.001)},
            pec=["ground", "trace"],
            # declared out of order on purpose
            ports={"pa": {"number": 2}, "pb": {"number": 1}},
            output_path=tmp_path / "out",
            num_points=11,
            show_structure=False,
            run=False,
        )

        ports = build_only["port"]
        assert isinstance(ports, list)
        assert [p.number for p in ports] == [1, 2]


# ---------------------------------------------------------------------
# simulate_step_FDTD -- against the real solver
# ---------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.needs_openems_bin
class TestSolverRun:
    """The ``run=True`` branch, against the real solver.

    Post-processed S-parameters are deliberately not asserted here, and the
    reason is a property of this prototype rather than of the test. The
    reconstructed geometry meshes down to sub-micron cells, which drives the
    FDTD timestep to ~3e-16 s and puts the Nyquist rate above 300,000
    timesteps. Any run short enough for a test suite therefore leaves a single
    sample in the port probes, and ``CalcPort`` cannot form a spectrum from
    one point. Reaching a usable spectrum needs a run in the minutes, which
    does not belong here.

    So what is checked is everything up to that: the solver accepts the
    imported model, runs it, and writes the files post-processing reads.
    """

    @pytest.fixture(scope="class")
    def ran(self, tmp_path_factory, request):
        asm = cq.Assembly(name="model")
        asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
        asm.add(box_at(-10, -10, -1.635, 20, 20, 0.035), name="ground")
        asm.add(box_at(-1.5, -8, 0, 3, 16, 0.035), name="trace")
        asm.add(box_at(-1.5, -8, -1.6, 3, 0.001, 1.6), name="p1")
        out = tmp_path_factory.mktemp("step_fdtd")
        step = out / "microstrip.step"
        asm.save(str(step))

        # Post-processing is stubbed for the reason in the class docstring;
        # monkeypatch is function-scoped, so patch by hand and undo at teardown.
        from simpleEMS import fdtd_import_step as module

        original = module.SimTools.compute_sim_data
        module.SimTools.compute_sim_data = staticmethod(
            lambda sim, port, output_path=None: None
        )
        request.addfinalizer(
            lambda: setattr(module.SimTools, "compute_sim_data", original)
        )

        simulate_step_FDTD(
            step_file=step,
            freqs=np.linspace(2e9, 3e9, 11),
            dielectrics={"substrate": (4.4, 0.001)},
            pec=["ground", "trace"],
            ports={"p1": {"direction": "z"}},
            output_path=out / "run",
            num_points=11,
            FDTD_timestep=300,
            show_structure=False,
            run=True,
        )
        return out / "run"

    def test_the_solver_wrote_the_port_probes(self, ran):
        """``compute_sim_data`` reads these by name."""
        assert (ran / "port_ut_1").is_file()
        assert (ran / "port_it_1").is_file()

    def test_the_probes_record_the_port_that_was_placed(self, ran):
        """openEMS writes the integration path into the file header, so this
        confirms the port landed on the marker solid's extents."""
        header = (ran / "port_ut_1").read_text()

        assert "start-coordinates" in header
        assert "stop-coordinates" in header

    def test_the_nf2ff_box_recorded_far_field_data(self, ran):
        """``create_nf2ff`` has to run before ``Run()`` or these are missing."""
        assert list(ran.glob("nf2ff_E_*.h5"))
        assert list(ran.glob("nf2ff_H_*.h5"))

    def test_the_structure_was_written(self, ran):
        assert (ran / "step_stl").is_dir()
