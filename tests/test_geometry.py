"""Geometry-construction tests for the structure builders.

These build real CSXCAD structures in-process and assert on the resulting
primitives -- no solver is ever invoked, so the whole file runs in a couple of
seconds. What is checked is the layout contract that ``sim_tools`` and the
exporters depend on: which properties exist, where each primitive sits, and
that the parts actually touch each other (feed into patch, ports at the line
ends, ground under the substrate).

Physics is *not* asserted here; that lives in ``test_calc.py``.
"""

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.ifa_antenna import (  # noqa: E402
    InvertedFAntenna,
    InvertedFAntennaParams,
)
from simpleEMS.microstrip_line import MicrostripLineParams  # noqa: E402
from simpleEMS.patch_antenna import (  # noqa: E402
    InsetFedPatchAntenna,
    InsetFedPatchParams,
    ProbeFedPatchAntenna,
)
from simpleEMS.quarterwave_stub_filter import (  # noqa: E402
    BandPassQuarterWaveFilter,
    BandStopQuarterWaveFilter,
    QuarterWaveFilterParams,
)
from simpleEMS.sim_tools import SimSetup, setup_simulation  # noqa: E402


def bbox(primitive) -> np.ndarray:
    """Return a primitive's bounding box as ``[[xmin, ymin, zmin], [xmax, ...]]``."""
    return np.array(primitive.GetBoundBox(), dtype=float)


def primitives_named(csx, name):
    """All primitives belonging to the property called ``name``."""
    return [p for p in csx.GetAllPrimitives() if p.GetProperty().GetName() == name]


def only(items):
    """Assert a single match and return it."""
    assert len(items) == 1, f"expected exactly one, got {len(items)}"
    return items[0]


def property_names(csx):
    return [prop.GetName() for prop in csx.GetAllProperties()]


@pytest.fixture
def spy_openems(monkeypatch):
    """Record the solver configuration ``setup_simulation`` applies.

    openEMS exposes ``SetBoundaryCond``/``SetGaussExcite`` but no matching
    getters, so the only way to assert on them is to capture the calls.
    """
    from openEMS.openEMS import openEMS

    class Recorder(openEMS):
        boundary = None
        excite = None

        def SetBoundaryCond(self, bc):  # noqa: N802 - matches the base class
            Recorder.boundary = list(bc)
            return super().SetBoundaryCond(bc)

        def SetGaussExcite(self, f0, fc):  # noqa: N802 - matches the base class
            Recorder.excite = (f0, fc)
            return super().SetGaussExcite(f0, fc)

    Recorder.boundary = None
    Recorder.excite = None
    monkeypatch.setattr("simpleEMS.sim_tools.openEMS", Recorder)
    return Recorder


# ---------------------------------------------------------------------
# setup_simulation
# ---------------------------------------------------------------------
class TestSetupSimulation:
    def test_returns_a_simsetup(self, inset_params):
        sim = setup_simulation(inset_params)

        assert isinstance(sim, SimSetup)

    def test_frequency_grid_spans_the_requested_range(self, inset_params):
        sim = setup_simulation(inset_params)
        fmin, fmax = inset_params.freq_range

        assert sim.freqs[0] == pytest.approx(fmin)
        assert sim.freqs[-1] == pytest.approx(fmax)
        assert len(sim.freqs) == inset_params.num_points

    def test_frequency_grid_is_uniform_and_increasing(self, inset_params):
        sim = setup_simulation(inset_params)
        steps = np.diff(sim.freqs)

        assert np.all(steps > 0)
        assert steps == pytest.approx(np.full(len(steps), steps[0]))

    def test_carries_the_backend_and_reference_impedance(self, inset_params):
        sim = setup_simulation(inset_params)

        assert sim.backend_engine == "FDTD"
        assert sim.charac_imp == inset_params.charac_imp

    def test_fdtd_backend_carries_no_fem_options(self, inset_params):
        """``FEM_options`` must be ``None`` for FDTD; a stray options object
        would send ``compute_sim_data`` down the FEM branch."""
        sim = setup_simulation(inset_params)

        assert sim.FEM_options is None

    def test_fem_backend_carries_the_bundled_options(self, fr4):
        params = InsetFedPatchParams(
            resonant_freq=2.45e9,
            span_freq=0.5e9,
            backend_engine="FEM",
            FEM_fe_order=2,
            **fr4,
        )

        sim = setup_simulation(params)

        assert sim.FEM_options is not None
        assert sim.FEM_options.fe_order == 2

    def test_default_boundaries_are_eight_cell_pml(self, inset_params, spy_openems):
        setup_simulation(inset_params)

        assert spy_openems.boundary == ["PML_8"] * 6

    def test_custom_boundaries_are_applied(self, inset_params, spy_openems):
        boundary = ["PMC", "PMC", "PEC", "PEC", "MUR", "MUR"]

        setup_simulation(inset_params, FDTD_boundary=boundary)

        assert spy_openems.boundary == boundary

    def test_excitation_is_centred_on_the_frequency_range(
        self, inset_params, spy_openems
    ):
        """The Gaussian pulse must cover exactly the requested band, or the
        S-parameters at the edges are excited too weakly to be meaningful."""
        setup_simulation(inset_params)

        fmin, fmax = inset_params.freq_range
        centre, span = spy_openems.excite

        assert centre == pytest.approx((fmin + fmax) / 2)
        assert span == pytest.approx((fmax - fmin) / 2)
        assert centre - span == pytest.approx(fmin)
        assert centre + span == pytest.approx(fmax)

    def test_csx_is_attached_to_the_fdtd_object(self, inset_params):
        """Geometry added via ``sim.CSX`` has to be the geometry the solver
        writes out."""
        sim = setup_simulation(inset_params)

        assert sim.FDTD.GetCSX() is sim.CSX

    def test_new_structure_starts_empty(self, inset_params):
        sim = setup_simulation(inset_params)

        assert sim.CSX.GetAllPrimitives() == []

    def test_each_call_builds_an_independent_structure(self, inset_params):
        first = setup_simulation(inset_params)
        second = setup_simulation(inset_params)

        assert first.CSX is not second.CSX


# ---------------------------------------------------------------------
# Inset-fed patch
# ---------------------------------------------------------------------
class TestInsetFedPatch:
    def test_builds_the_expected_properties(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        names = property_names(sim.CSX)

        assert "substrate" in names
        assert "ground" in names
        assert "patch_inset" in names
        assert "feed" in names

    def test_patch_is_a_polygon_not_a_box(self, built_inset):
        """The inset notch makes the patch non-rectangular; emitting a plain
        box would silently drop the notch."""
        _antenna, sim, _params, _port = built_inset

        patch = only(primitives_named(sim.CSX, "patch_inset"))

        assert patch.__class__.__name__ == "CSPrimLinPoly"

    def test_patch_bounding_box_matches_the_computed_dimensions(self, built_inset):
        _antenna, sim, params, _port = built_inset

        box = bbox(only(primitives_named(sim.CSX, "patch_inset")))

        assert box[1][0] - box[0][0] == pytest.approx(params.patch_width_mm, abs=1e-3)
        assert box[1][1] - box[0][1] == pytest.approx(params.patch_length_mm, abs=1e-3)

    def test_patch_is_centred_on_the_origin(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        box = bbox(only(primitives_named(sim.CSX, "patch_inset")))

        assert (box[0][0] + box[1][0]) / 2 == pytest.approx(0.0, abs=1e-6)
        assert (box[0][1] + box[1][1]) / 2 == pytest.approx(0.0, abs=1e-6)

    def test_patch_sits_on_top_of_the_substrate(self, built_inset):
        _antenna, sim, params, _port = built_inset

        patch = bbox(only(primitives_named(sim.CSX, "patch_inset")))

        assert patch[0][2] == pytest.approx(params.substrate_thickness_mm, abs=1e-6)
        assert patch[1][2] - patch[0][2] == pytest.approx(
            params.copper_thickness_mm, abs=1e-6
        )

    def test_feed_width_matches_the_computed_feed_width(self, built_inset):
        _antenna, sim, params, _port = built_inset

        feed = bbox(only(primitives_named(sim.CSX, "feed")))

        assert feed[1][0] - feed[0][0] == pytest.approx(params.feed_width_mm, abs=1e-3)

    def test_feed_penetrates_the_patch_by_the_inset_depth(self, built_inset):
        """The single most important connectivity invariant: a feed that stops
        short of the patch leaves an open circuit, and the simulation still
        runs and produces an S11 of ~0 dB."""
        _antenna, sim, params, _port = built_inset

        patch = bbox(only(primitives_named(sim.CSX, "patch_inset")))
        feed = bbox(only(primitives_named(sim.CSX, "feed")))

        penetration = feed[1][1] - patch[0][1]

        assert penetration == pytest.approx(params.inset_length_mm, abs=1e-2)

    def test_feed_is_coplanar_with_the_patch(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        patch = bbox(only(primitives_named(sim.CSX, "patch_inset")))
        feed = bbox(only(primitives_named(sim.CSX, "feed")))

        assert feed[0][2] == pytest.approx(patch[0][2], abs=1e-9)
        assert feed[1][2] == pytest.approx(patch[1][2], abs=1e-9)

    def test_feed_is_centred_in_x(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        feed = bbox(only(primitives_named(sim.CSX, "feed")))

        assert (feed[0][0] + feed[1][0]) / 2 == pytest.approx(0.0, abs=1e-6)

    def test_substrate_encloses_the_patch(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        substrate = bbox(only(primitives_named(sim.CSX, "substrate")))
        patch = bbox(only(primitives_named(sim.CSX, "patch_inset")))

        assert substrate[0][0] <= patch[0][0]
        assert substrate[1][0] >= patch[1][0]
        assert substrate[0][1] <= patch[0][1]
        assert substrate[1][1] >= patch[1][1]

    def test_substrate_matches_the_declared_dimensions(self, built_inset):
        _antenna, sim, params, _port = built_inset

        box = bbox(only(primitives_named(sim.CSX, "substrate")))

        assert box[1][0] - box[0][0] == pytest.approx(
            params.substrate_width_mm, abs=1e-3
        )
        assert box[1][1] - box[0][1] == pytest.approx(
            params.substrate_length_mm, abs=1e-3
        )
        assert box[1][2] - box[0][2] == pytest.approx(
            params.substrate_thickness_mm, abs=1e-6
        )

    def test_ground_sits_directly_beneath_the_substrate(self, built_inset):
        _antenna, sim, params, _port = built_inset

        substrate = bbox(only(primitives_named(sim.CSX, "substrate")))
        ground = bbox(only(primitives_named(sim.CSX, "ground")))

        assert ground[1][2] == pytest.approx(substrate[0][2], abs=1e-9)
        assert ground[1][2] - ground[0][2] == pytest.approx(
            params.copper_thickness_mm, abs=1e-6
        )

    def test_ground_spans_the_whole_substrate(self, built_inset):
        _antenna, sim, _params, _port = built_inset

        substrate = bbox(only(primitives_named(sim.CSX, "substrate")))
        ground = bbox(only(primitives_named(sim.CSX, "ground")))

        assert ground[0][:2] == pytest.approx(substrate[0][:2], abs=1e-6)
        assert ground[1][:2] == pytest.approx(substrate[1][:2], abs=1e-6)

    def test_substrate_carries_the_computed_permittivity_and_loss(self, built_inset):
        _antenna, sim, params, _port = built_inset

        substrate = only(
            [p for p in sim.CSX.GetAllProperties() if p.GetName() == "substrate"]
        )

        assert substrate.GetMaterialProperty("epsilon") == pytest.approx(
            params.substrate_eps_r
        )
        assert substrate.GetMaterialProperty("kappa") == pytest.approx(
            params.substrate_kappa
        )

    def test_metal_outranks_substrate_in_priority(self, built_inset):
        """Overlapping primitives are resolved by priority; a patch that does
        not outrank the substrate would be swallowed by the dielectric."""
        _antenna, sim, _params, _port = built_inset

        substrate = only(primitives_named(sim.CSX, "substrate"))
        feed = only(primitives_named(sim.CSX, "feed"))
        ground = only(primitives_named(sim.CSX, "ground"))

        assert feed.GetPriority() > substrate.GetPriority()
        assert ground.GetPriority() > substrate.GetPriority()

    def test_single_excited_port_is_returned(self, built_inset):
        _antenna, _sim, _params, port = built_inset

        assert port.number == 1
        assert port.excite == 1

    def test_port_sits_at_the_open_end_of_the_feed(self, built_inset):
        _antenna, sim, _params, port = built_inset

        feed = bbox(only(primitives_named(sim.CSX, "feed")))

        assert port.start[1] == pytest.approx(feed[0][1], abs=1e-6)

    def test_port_spans_the_feed_width(self, built_inset):
        _antenna, sim, params, port = built_inset

        assert abs(port.start[0] - port.stop[0]) == pytest.approx(
            params.feed_width_mm, abs=1e-3
        )

    def test_port_bridges_ground_to_trace_in_z(self, built_inset):
        """A lumped port must span the full dielectric stack, ground plane to
        trace, or it excites nothing."""
        _antenna, sim, _params, port = built_inset

        ground = bbox(only(primitives_named(sim.CSX, "ground")))
        feed = bbox(only(primitives_named(sim.CSX, "feed")))
        z_low, z_high = sorted([port.start[2], port.stop[2]])

        assert z_low == pytest.approx(ground[0][2], abs=1e-6)
        assert z_high == pytest.approx(feed[1][2], abs=1e-6)

    def test_geometry_fits_inside_the_simulation_box(self, built_inset):
        _antenna, sim, params, _port = built_inset

        box = params._default_simulation_box
        for primitive in sim.CSX.GetAllPrimitives():
            limits = bbox(primitive)
            assert limits[0][0] >= -box[0] / 2 - 1e-6
            assert limits[1][0] <= box[0] / 2 + 1e-6
            assert limits[0][1] >= -box[1] / 2 - 1e-6
            assert limits[1][1] <= box[1] / 2 + 1e-6

    def test_rebuilding_is_deterministic(self, inset_params):
        """Two builds from the same params must be geometrically identical."""
        boxes = []
        for _ in range(2):
            sim = setup_simulation(inset_params)
            InsetFedPatchAntenna(inset_params, sim).build_inset_fed_patch_antenna()
            boxes.append([bbox(p).tolist() for p in sim.CSX.GetAllPrimitives()])

        assert boxes[0] == boxes[1]


# ---------------------------------------------------------------------
# Inset-fed patch parameter validation
# ---------------------------------------------------------------------
class TestInsetFedPatchParams:
    def test_frequency_range_brackets_the_resonance(self, inset_params):
        fmin, fmax = inset_params.freq_range

        assert fmin < inset_params.resonant_freq < fmax
        assert fmax - fmin == pytest.approx(2 * inset_params.span_freq)

    def test_main_freq_is_the_resonant_frequency(self, inset_params):
        assert inset_params.main_freq == inset_params.resonant_freq

    def test_substrate_adds_a_wavelength_of_margin(self, inset_params):
        assert inset_params.substrate_width_mm == pytest.approx(
            inset_params.patch_width_mm + 2 * inset_params.lambda0
        )
        assert inset_params.substrate_length_mm == pytest.approx(
            inset_params.patch_length_mm + 2 * inset_params.lambda0
        )

    def test_inset_width_is_half_the_feed_line_width(self, inset_params):
        """``_compute_geometry`` halves the computed line width to get each
        of the two flanking slots."""
        assert inset_params.inset_width_mm == pytest.approx(
            inset_params.feed_width_mm / 2, abs=1e-3
        )

    def test_geometry_is_rounded_to_fp_precision(self, inset_params):
        for name in (
            "patch_width_mm",
            "patch_length_mm",
            "inset_width_mm",
            "inset_length_mm",
            "feed_width_mm",
            "feed_length_mm",
        ):
            value = getattr(inset_params, name)
            assert value == pytest.approx(round(value, inset_params.fp_precision))

    def test_too_narrow_an_inset_gap_is_rejected(self, fr4):
        """Guards against emitting a layout the fab house cannot etch."""
        params = {**fr4, "min_trace_spacing_mm": 50.0}

        with pytest.raises(ValueError, match="Inset gap"):
            InsetFedPatchParams(resonant_freq=2.45e9, span_freq=0.5e9, **params)

    def test_unrealizable_feed_impedance_is_rejected(self, fr4):
        params = {**fr4, "charac_imp": 5000}

        with pytest.raises(ValueError):
            InsetFedPatchParams(resonant_freq=2.45e9, span_freq=0.5e9, **params)


# ---------------------------------------------------------------------
# Probe-fed patch
# ---------------------------------------------------------------------
class TestProbeFedPatch:
    @pytest.fixture
    def built(self, probe_params):
        sim = setup_simulation(probe_params)
        antenna = ProbeFedPatchAntenna(probe_params, sim)
        port = antenna.build_probe_fed_patch_antenna()
        return antenna, sim, probe_params, port

    def test_builds_the_expected_properties(self, built):
        _antenna, sim, _params, _port = built

        names = property_names(sim.CSX)

        assert "substrate" in names
        assert "ground" in names
        assert "patch_probe" in names

    def test_patch_is_a_plain_rectangle(self, built):
        """No inset notch on a probe feed, so a box is the right primitive."""
        _antenna, sim, _params, _port = built

        patch = only(primitives_named(sim.CSX, "patch_probe"))

        assert patch.__class__.__name__ == "CSPrimBox"

    def test_has_no_feed_line(self, built):
        _antenna, sim, _params, _port = built

        assert "feed" not in property_names(sim.CSX)

    def test_patch_dimensions_match_the_params(self, built):
        _antenna, sim, params, _port = built

        box = bbox(only(primitives_named(sim.CSX, "patch_probe")))

        assert box[1][0] - box[0][0] == pytest.approx(params.patch_width_mm, abs=1e-3)
        assert box[1][1] - box[0][1] == pytest.approx(params.patch_length_mm, abs=1e-3)

    def test_probe_is_at_the_computed_feed_position(self, built):
        _antenna, _sim, params, port = built

        assert port.start[1] == pytest.approx(params.probe_pos_mm, abs=1e-3)

    def test_probe_position_lies_within_the_patch(self, built):
        _antenna, sim, _params, port = built

        patch = bbox(only(primitives_named(sim.CSX, "patch_probe")))

        assert patch[0][1] <= port.start[1] <= patch[1][1]

    def test_probe_spans_the_dielectric_only(self, built):
        """The probe pin runs from the ground plane up to the patch, through
        the substrate."""
        _antenna, sim, params, port = built

        substrate = bbox(only(primitives_named(sim.CSX, "substrate")))
        z_low, z_high = sorted([port.start[2], port.stop[2]])

        assert z_low == pytest.approx(substrate[0][2], abs=1e-6)
        assert z_high == pytest.approx(params.substrate_thickness_mm, abs=1e-6)

    def test_single_excited_port(self, built):
        _antenna, _sim, _params, port = built

        assert port.number == 1
        assert port.excite == 1


# ---------------------------------------------------------------------
# Microstrip line
# ---------------------------------------------------------------------
class TestMicrostripLine:
    def test_builds_the_expected_properties(self, built_mline):
        _line, sim, _params, _ports = built_mline

        names = property_names(sim.CSX)

        assert "substrate" in names
        assert "ground" in names
        assert "microstrip" in names

    def test_trace_dimensions_match_the_params(self, built_mline):
        _line, sim, params, _ports = built_mline

        box = bbox(only(primitives_named(sim.CSX, "microstrip")))

        assert box[1][0] - box[0][0] == pytest.approx(
            params.microstrip_width_mm, abs=1e-3
        )
        assert box[1][1] - box[0][1] == pytest.approx(
            params.microstrip_length_mm, abs=1e-3
        )

    def test_trace_is_centred(self, built_mline):
        _line, sim, _params, _ports = built_mline

        box = bbox(only(primitives_named(sim.CSX, "microstrip")))

        assert (box[0][0] + box[1][0]) / 2 == pytest.approx(0.0, abs=1e-9)
        assert (box[0][1] + box[1][1]) / 2 == pytest.approx(0.0, abs=1e-9)

    def test_returns_two_ports(self, built_mline):
        _line, _sim, _params, ports = built_mline

        assert [port.number for port in ports] == [1, 2]

    def test_only_the_first_port_is_excited(self, built_mline):
        """S21 is meaningless if both ports drive the structure."""
        _line, _sim, _params, ports = built_mline

        assert ports[0].excite == 1
        assert ports[1].excite == 0

    def test_ports_sit_at_opposite_ends_of_the_trace(self, built_mline):
        _line, sim, _params, ports = built_mline

        trace = bbox(only(primitives_named(sim.CSX, "microstrip")))

        assert ports[0].start[1] == pytest.approx(trace[0][1], abs=1e-6)
        assert ports[1].start[1] == pytest.approx(trace[1][1], abs=1e-6)

    def test_ports_sit_on_the_board_edges(self, built_mline):
        """The trace runs edge to edge, so both connectors land on the board
        edges rather than in the middle of a dielectric apron."""
        _line, _sim, params, ports = built_mline

        assert ports[0].start[1] == pytest.approx(
            -params.substrate_length_mm / 2, abs=1e-6
        )
        assert ports[1].start[1] == pytest.approx(
            params.substrate_length_mm / 2, abs=1e-6
        )

    def test_ports_are_separated_by_the_trace_length(self, built_mline):
        _line, _sim, params, ports = built_mline

        separation = abs(ports[1].start[1] - ports[0].start[1])

        assert separation == pytest.approx(params.microstrip_length_mm, abs=1e-3)

    def test_quarter_wave_length_matches_the_electrical_length(self, mline_params):
        """The default is 90 degrees, so the line is a quarter wavelength."""
        assert mline_params.elec_length_deg == 90
        assert mline_params.microstrip_length_mm > 0

    def test_substrate_width_pads_the_trace_by_six_thicknesses(self, mline_params):
        assert mline_params.substrate_width_mm == pytest.approx(
            mline_params.microstrip_width_mm + 12 * mline_params.substrate_thickness_mm
        )

    def test_substrate_length_is_the_trace_length(self, mline_params):
        """The trace runs edge to edge, so both ports land on board edges."""
        assert mline_params.substrate_length_mm == pytest.approx(
            mline_params.microstrip_length_mm
        )

    def test_too_narrow_a_trace_is_rejected(self, fr4):
        params = {**fr4, "min_trace_width_mm": 50.0}

        with pytest.raises(ValueError, match="Microstrip width"):
            MicrostripLineParams(
                min_freq=1e9, max_freq=4e9, target_freq=2.45e9, **params
            )

    def test_main_freq_is_the_target_frequency(self, mline_params):
        assert mline_params.main_freq == mline_params.target_freq
        assert mline_params.freq_range == (
            mline_params.min_freq,
            mline_params.max_freq,
        )


# ---------------------------------------------------------------------
# Quarter-wave stub filter
# ---------------------------------------------------------------------
class TestQuarterWaveFilter:
    @pytest.fixture
    def built_bandstop(self, filter_params):
        sim = setup_simulation(filter_params)
        structure = BandStopQuarterWaveFilter(filter_params, sim)
        ports = structure.build_band_stop_quarter_wave_filter()
        return structure, sim, filter_params, ports

    def test_emits_one_series_line_per_section(self, built_bandstop):
        """An order-n filter has n stubs and n + 1 series sections."""
        _structure, sim, params, _ports = built_bandstop

        names = property_names(sim.CSX)
        series = [n for n in names if n.startswith("series_line_")]
        shunt = [n for n in names if n.startswith("shunt_line_")]

        assert len(series) == params.filter_order + 1
        assert len(shunt) == params.filter_order

    def test_stub_widths_match_the_computed_values(self, built_bandstop):
        _structure, sim, params, _ports = built_bandstop

        for index, expected in enumerate(params.shunt_line_width_mm, start=1):
            box = bbox(only(primitives_named(sim.CSX, f"shunt_line_{index}")))
            assert box[1][0] - box[0][0] == pytest.approx(expected, abs=1e-2)

    def test_series_sections_share_the_line_width(self, built_bandstop):
        _structure, sim, params, _ports = built_bandstop

        for index in range(1, params.filter_order + 2):
            box = bbox(only(primitives_named(sim.CSX, f"series_line_{index}")))
            assert box[1][1] - box[0][1] == pytest.approx(
                params.series_line_width_mm, abs=1e-2
            )

    def test_sections_are_laid_out_left_to_right_without_gaps(self, built_bandstop):
        """Each series section must butt against the next stub; a gap is an
        open circuit that still simulates."""
        _structure, sim, params, _ports = built_bandstop

        for index in range(1, params.filter_order + 1):
            series = bbox(only(primitives_named(sim.CSX, f"series_line_{index}")))
            stub = bbox(only(primitives_named(sim.CSX, f"shunt_line_{index}")))
            next_series = bbox(
                only(primitives_named(sim.CSX, f"series_line_{index + 1}"))
            )

            assert stub[0][0] == pytest.approx(series[1][0], abs=1e-2)
            assert next_series[0][0] == pytest.approx(stub[1][0], abs=1e-2)

    def test_stubs_extend_away_from_the_series_line(self, built_bandstop):
        _structure, sim, params, _ports = built_bandstop

        for index in range(1, params.filter_order + 1):
            stub = bbox(only(primitives_named(sim.CSX, f"shunt_line_{index}")))
            assert stub[1][1] - stub[0][1] > params.series_line_width_mm

    def test_all_metal_is_coplanar(self, built_bandstop):
        _structure, sim, params, _ports = built_bandstop

        top = params.substrate_thickness_mm
        for name in property_names(sim.CSX):
            if name.startswith(("series_line_", "shunt_line_")):
                box = bbox(only(primitives_named(sim.CSX, name)))
                assert box[0][2] == pytest.approx(top, abs=1e-6)

    def test_returns_two_ports_with_one_excited(self, built_bandstop):
        _structure, _sim, _params, ports = built_bandstop

        assert [port.number for port in ports] == [1, 2]
        assert ports[0].excite == 1
        assert ports[1].excite == 0

    def test_ports_sit_at_the_ends_of_the_series_chain(self, built_bandstop):
        _structure, sim, params, ports = built_bandstop

        first = bbox(only(primitives_named(sim.CSX, "series_line_1")))
        last = bbox(
            only(primitives_named(sim.CSX, f"series_line_{params.filter_order + 1}"))
        )

        assert ports[0].start[0] == pytest.approx(first[0][0], abs=1e-2)
        assert ports[1].start[0] == pytest.approx(last[1][0], abs=1e-2)

    def test_bandpass_variant_builds(self):
        """Parameters taken from ``examples/BandPassQuarterWaveFilter.py``; a
        band-pass design needs low-impedance (wide) stubs, so the band-stop
        fixture's substrate would fail the stub-overlap check."""
        params = QuarterWaveFilterParams(
            substrate_eps_r=3.3,
            substrate_tand=0.001,
            substrate_thickness_mm=1.6,
            min_freq=0.5e9,
            max_freq=3e9,
            centre_freq=1.5e9,
            bandwidth_freq=1e9,
            filter_type="bandpass",
            filter_response="butterworth",
            filter_order=3,
        )
        sim = setup_simulation(params)

        ports = BandPassQuarterWaveFilter(
            params, sim
        ).build_band_pass_quarter_wave_filter()

        assert len(ports) == 2
        assert len(sim.CSX.GetAllPrimitives()) > 0

    @pytest.mark.parametrize("filter_type", ["lowpass", "highpass", "notch", ""])
    def test_unsupported_filter_type_is_rejected(self, fr4, filter_type):
        with pytest.raises(ValueError, match="does not support"):
            QuarterWaveFilterParams(
                min_freq=1e9,
                max_freq=4e9,
                centre_freq=2.45e9,
                bandwidth_freq=1.0e9,
                filter_type=filter_type,
                filter_response="butterworth",
                filter_order=3,
                **fr4,
            )

    def test_fractional_bandwidth_is_bandwidth_over_centre(self, filter_params):
        assert filter_params.frac_bandwidth == pytest.approx(
            filter_params.bandwidth_freq / filter_params.centre_freq, abs=1e-3
        )

    @pytest.mark.parametrize("bandwidth", [0.0, -1e9])
    def test_non_positive_bandwidth_is_rejected(self, fr4, bandwidth):
        with pytest.raises(ValueError):
            QuarterWaveFilterParams(
                min_freq=1e9,
                max_freq=4e9,
                centre_freq=2.45e9,
                bandwidth_freq=bandwidth,
                filter_type="bandstop",
                filter_response="butterworth",
                filter_order=3,
                **fr4,
            )

    def test_stub_count_matches_the_order(self, filter_params):
        assert len(filter_params.shunt_line_width_mm) == filter_params.filter_order
        assert len(filter_params.shunt_line_length_mm) == filter_params.filter_order

    def test_butterworth_stubs_are_symmetric(self, filter_params):
        """A Butterworth prototype is a palindrome, so the outer stubs must
        come out identical."""
        widths = filter_params.shunt_line_width_mm

        assert widths[0] == pytest.approx(widths[-1], abs=1e-3)


# ---------------------------------------------------------------------
# Automatic mesh generation
# ---------------------------------------------------------------------
class TestMeshGeneration:
    @pytest.fixture
    def meshed_inset(self, built_inset):
        antenna, sim, params, _port = built_inset
        antenna.create_mesh()
        return sim, params

    def lines(self, sim):
        grid = sim.CSX.GetGrid()
        return [np.asarray(grid.GetLines(dim)) for dim in range(3)]

    def test_every_dimension_gets_mesh_lines(self, meshed_inset):
        sim, _params = meshed_inset

        for axis in self.lines(sim):
            assert len(axis) > 1

    def test_lines_are_sorted_and_unique(self, meshed_inset):
        sim, _params = meshed_inset

        for axis in self.lines(sim):
            assert np.all(np.diff(axis) > 0)

    def test_cell_size_respects_the_global_resolution(self, meshed_inset):
        """The whole point of ``FDTD_mesh_resolution``: no cell may exceed it,
        or the grid under-samples the wave."""
        sim, params = meshed_inset

        for axis in self.lines(sim):
            assert np.max(np.diff(axis)) <= params.FDTD_mesh_resolution * 1.05

    def test_mesh_covers_the_structure(self, meshed_inset):
        sim, _params = meshed_inset

        axes = self.lines(sim)
        for primitive in sim.CSX.GetAllPrimitives():
            limits = bbox(primitive)
            for dim in range(3):
                assert axes[dim][0] <= limits[0][dim] + 1e-6
                assert axes[dim][-1] >= limits[1][dim] - 1e-6

    def test_metal_edges_get_nearby_mesh_lines(self, meshed_inset):
        """The thirds rule places lines close to every conductor edge; without
        them the FDTD field at the edge is badly resolved."""
        sim, params = meshed_inset

        axes = self.lines(sim)
        patch = bbox(only(primitives_named(sim.CSX, "patch_inset")))

        for dim in (0, 1):
            for edge in (patch[0][dim], patch[1][dim]):
                nearest = np.min(np.abs(axes[dim] - edge))
                assert nearest <= params.FDTD_mesh_resolution

    def test_substrate_thickness_is_resolved_by_several_cells(self, meshed_inset):
        sim, params = meshed_inset

        z_lines = self.lines(sim)[2]
        inside = z_lines[
            (z_lines >= -1e-9) & (z_lines <= params.substrate_thickness_mm + 1e-9)
        ]

        assert len(inside) >= 2

    def test_mesh_is_deterministic(self, inset_params):
        first = None
        for _ in range(2):
            sim = setup_simulation(inset_params)
            antenna = InsetFedPatchAntenna(inset_params, sim)
            antenna.build_inset_fed_patch_antenna()
            antenna.create_mesh()
            current = [np.asarray(sim.CSX.GetGrid().GetLines(d)) for d in range(3)]
            if first is None:
                first = current
            else:
                for a, b in zip(first, current, strict=True):
                    assert a == pytest.approx(b)

    def test_grid_delta_unit_is_set(self, meshed_inset):
        """openEMS interprets every coordinate through this scale factor."""
        sim, params = meshed_inset

        assert sim.CSX.GetGrid().GetDeltaUnit() == pytest.approx(params.unit)

    def test_manual_mesh_path_also_produces_a_grid(self, built_mline):
        """``manual_mesh=True`` is a separate, hand-rolled code path."""
        line, sim, params, _ports = built_mline

        line.create_mesh(manual_mesh=True)

        for dim in range(3):
            axis = np.asarray(sim.CSX.GetGrid().GetLines(dim))
            assert len(axis) > 1
            assert np.all(np.diff(axis) > 0)
            assert np.max(np.diff(axis)) <= params.FDTD_mesh_resolution * 1.6

    def test_microstrip_auto_mesh(self, built_mline):
        line, sim, params, _ports = built_mline

        line.create_mesh()

        for dim in range(3):
            axis = np.asarray(sim.CSX.GetGrid().GetLines(dim))
            assert len(axis) > 1
            assert np.max(np.diff(axis)) <= params.FDTD_mesh_resolution * 1.05


# ---------------------------------------------------------------------
# create_mesh across every builder
# ---------------------------------------------------------------------
class TestMeshAcrossStructures:
    """``create_mesh`` on every structure class, in both of its code paths.

    ``TestMeshGeneration`` above pins the grid contract in detail, but only
    for the inset patch. Each builder re-implements ``create_mesh`` from
    scratch -- and each implementation has a separate hand-rolled
    ``manual_mesh=True`` branch that shares no code with the automatic one --
    so the same contract has to be checked against all of them.
    """

    STRUCTURES = ["inset", "probe", "mline", "bandstop", "bandpass"]

    @pytest.fixture(params=STRUCTURES)
    def built(self, request):
        """Every structure class, one per parametrisation."""
        structure, sim, params, _ports = request.getfixturevalue(
            f"built_{request.param}"
        )
        return structure, sim, params

    @staticmethod
    def lines(sim):
        grid = sim.CSX.GetGrid()
        return [np.asarray(grid.GetLines(dim)) for dim in range(3)]

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_every_dimension_gets_a_sorted_unique_grid(self, built, manual):
        structure, sim, _params = built

        structure.create_mesh(manual_mesh=manual)

        for axis in self.lines(sim):
            assert len(axis) > 1
            assert np.all(np.diff(axis) > 0)

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_grid_delta_unit_is_set(self, built, manual):
        structure, sim, params = built

        structure.create_mesh(manual_mesh=manual)

        assert sim.CSX.GetGrid().GetDeltaUnit() == pytest.approx(params.unit)

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_mesh_covers_every_primitive(self, built, manual):
        """A primitive outside the grid is silently not simulated."""
        structure, sim, _params = built

        structure.create_mesh(manual_mesh=manual)

        axes = self.lines(sim)
        for primitive in sim.CSX.GetAllPrimitives():
            limits = bbox(primitive)
            for dim in range(3):
                assert axes[dim][0] <= limits[0][dim] + 1e-6
                assert axes[dim][-1] >= limits[1][dim] - 1e-6

    def test_auto_mesh_respects_the_global_resolution(self, built):
        """No cell may exceed ``FDTD_mesh_resolution`` or the grid
        under-samples the wave. Only the automatic path promises this; the
        manual path hand-places its lines and is checked more loosely below.
        """
        structure, sim, params = built

        structure.create_mesh()

        for axis in self.lines(sim):
            assert np.max(np.diff(axis)) <= params.FDTD_mesh_resolution * 1.05

    def test_manual_mesh_stays_near_the_global_resolution(self, built):
        structure, sim, params = built

        structure.create_mesh(manual_mesh=True)

        for axis in self.lines(sim):
            assert np.max(np.diff(axis)) <= params.FDTD_mesh_resolution * 1.6

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_substrate_thickness_is_resolved(self, built, manual):
        structure, sim, params = built

        structure.create_mesh(manual_mesh=manual)

        z_lines = self.lines(sim)[2]
        inside = z_lines[
            (z_lines >= -1e-9) & (z_lines <= params.substrate_thickness_mm + 1e-9)
        ]

        assert len(inside) >= 2

    @staticmethod
    def substrate_interior_lines(sim, params):
        z_lines = np.asarray(sim.CSX.GetGrid().GetLines(2))
        tolerance = 1e-6
        return z_lines[
            (z_lines > tolerance)
            & (z_lines < params.substrate_thickness_mm - tolerance)
        ]

    def test_auto_mesh_puts_substrate_cells_lines_through_the_substrate(self, built):
        """``substrate_cells`` counts both faces; the interior lines are evenly
        spaced whatever the frequency, so a thin substrate at a long wavelength
        is not collapsed to one midpoint line."""
        structure, sim, params = built

        structure.create_mesh()

        expected = np.linspace(
            0.0, params.substrate_thickness_mm, params.substrate_cells
        )[1:-1]
        interior_lines = self.substrate_interior_lines(sim, params)
        assert interior_lines == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_meshing_twice_is_idempotent(self, built, manual):
        """``create_mesh`` is called once per run, but re-running it must not
        accumulate duplicate lines -- the grid is shared mutable state."""
        structure, sim, _params = built

        structure.create_mesh(manual_mesh=manual)
        first = self.lines(sim)
        structure.create_mesh(manual_mesh=manual)
        second = self.lines(sim)

        for a, b in zip(first, second, strict=True):
            assert a == pytest.approx(b)


# ---------------------------------------------------------------------
# User-defined simulation box
# ---------------------------------------------------------------------
class TestUserSimulationBox:
    """``create_mesh`` honours ``params.simulation_box`` when it is defined and
    derives the box from the geometry only when it is ``None``."""

    @pytest.fixture
    def build_line(self, mline_params, sim_for):
        """Factory building a microstrip line whose params carry a given box."""
        from simpleEMS.microstrip_line import MicrostripLine

        def _build(simulation_box):
            params = dataclasses.replace(mline_params, simulation_box=simulation_box)
            simulation = sim_for(params)
            line = MicrostripLine(params, simulation)
            line.build_microstrip_line()
            return line, simulation, params

        return _build

    @staticmethod
    def outer_lines(simulation):
        grid = simulation.CSX.GetGrid()
        return np.array(
            [
                [grid.GetLines(dimension)[0], grid.GetLines(dimension)[-1]]
                for dimension in range(3)
            ]
        )

    @staticmethod
    def sizes_around_the_substrate(params):
        return [
            params.substrate_width_mm + 40.0,
            params.substrate_length_mm + 40.0,
            40.0,
        ]

    def test_no_box_pads_the_geometry(self, build_line):
        from simpleEMS.fdtd_mesh import auto_simulation_bounds, geometry_extent

        line, simulation, params = build_line(None)
        expected = auto_simulation_bounds(
            geometry_extent(simulation.CSX), params.lambda0
        )

        line.create_mesh()

        assert self.outer_lines(simulation) == pytest.approx(
            np.array(expected), abs=1e-3
        )

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_sizes_box_sets_the_outer_grid_lines(
        self, build_line, mline_params, manual
    ):
        box_sizes = self.sizes_around_the_substrate(mline_params)
        line, simulation, _params = build_line(box_sizes)

        line.create_mesh(manual_mesh=manual)

        half_sizes = np.array(box_sizes) / 2
        assert self.outer_lines(simulation) == pytest.approx(
            np.column_stack((-half_sizes, half_sizes)), abs=1e-6
        )

    @pytest.mark.parametrize("manual", [False, True], ids=["auto", "manual"])
    def test_off_centre_bounds_set_the_outer_grid_lines(
        self, build_line, mline_params, manual
    ):
        half_width = mline_params.substrate_width_mm / 2
        half_length = mline_params.substrate_length_mm / 2
        bounds = [
            [-half_width - 10.0, half_width + 30.0],
            [-half_length - 20.0, half_length + 5.0],
            [-5.0, 25.0],
        ]
        line, simulation, _params = build_line(bounds)

        line.create_mesh(manual_mesh=manual)

        assert self.outer_lines(simulation) == pytest.approx(np.array(bounds), abs=1e-6)

    def test_box_smaller_than_the_structure_raises(self, build_line):
        line, _simulation, _params = build_line([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="too small"):
            line.create_mesh()


# ---------------------------------------------------------------------
# Substrate z-lines follow substrate_cells
# ---------------------------------------------------------------------
class TestSubstrateCells:
    @staticmethod
    def substrate_interior_lines(simulation, params):
        z_lines = np.asarray(simulation.CSX.GetGrid().GetLines(2))
        tolerance = 1e-6
        return z_lines[
            (z_lines > tolerance)
            & (z_lines < params.substrate_thickness_mm - tolerance)
        ]

    @pytest.fixture
    def build_bandpass(self, bandpass_filter_params, sim_for):
        """Factory building the band-pass filter with a given ``substrate_cells``.

        The band-pass filter is the case the thin-interval collapse used to
        reduce to one line: its substrate is under ``FDTD_mesh_resolution / 4``.
        """

        def _build(substrate_cells):
            params = dataclasses.replace(
                bandpass_filter_params, substrate_cells=substrate_cells
            )
            simulation = sim_for(params)
            structure = BandPassQuarterWaveFilter(params, simulation)
            structure.build_band_pass_quarter_wave_filter()
            return structure, simulation, params

        return _build

    @pytest.mark.parametrize("substrate_cells", [2, 4, 10], ids=["two", "four", "ten"])
    def test_interior_line_count_follows_the_parameter(
        self, build_bandpass, substrate_cells
    ):
        structure, simulation, params = build_bandpass(substrate_cells)

        structure.create_mesh()

        expected = np.linspace(0.0, params.substrate_thickness_mm, substrate_cells)[
            1:-1
        ]
        assert self.substrate_interior_lines(simulation, params) == pytest.approx(
            expected, abs=1e-3
        )

    def test_requested_line_inside_the_substrate_is_kept(self, build_bandpass):
        """``requested_lines`` still places its line, and the substrate still
        gets its evenly spaced lines around it."""
        from simpleEMS.fdtd_mesh import Mesh

        _structure, simulation, params = build_bandpass(7)
        requested_position = 0.5

        Mesh(simulation.CSX, params, requested_lines=[[], [], [requested_position]])

        expected = np.linspace(0.0, params.substrate_thickness_mm, 7)[1:-1]
        interior_lines = self.substrate_interior_lines(simulation, params)
        assert np.min(np.abs(interior_lines - requested_position)) < 1e-6
        for position in expected:
            assert np.min(np.abs(interior_lines - position)) < 1e-3
        assert len(interior_lines) == len(expected) + 1


# ---------------------------------------------------------------------
# Copper layers get one z-line whatever the frequency
# ---------------------------------------------------------------------
class TestCopperLayerZLines:
    """A copper layer is one line thick in z at any design frequency.

    ``FDTD_metal_mesh_resolution`` shrinks with the wavelength while
    ``copper_thickness_mm`` stays at 35 um, so above roughly 40 GHz the foil
    stops being thin relative to the metal resolution. The thin-interval
    collapse in ``_gen_mesh_in_bounds`` has to keep collapsing it anyway --
    otherwise the layer takes ``min_lines`` lines across 35 um, a spacing far
    finer than anything else in the grid, and the FDTD timestep follows the
    smallest cell.
    """

    @staticmethod
    def lines_within(simulation, lower, upper):
        z_lines = np.asarray(simulation.CSX.GetGrid().GetLines(2))
        return z_lines[(z_lines >= lower - 1e-9) & (z_lines <= upper + 1e-9)]

    @pytest.mark.parametrize(
        ("resonant_freq", "substrate_thickness_mm"),
        [(2.45e9, 1.6), (24e9, 0.254), (60e9, 0.125)],
        ids=["2.45GHz", "24GHz", "60GHz"],
    )
    def test_each_copper_layer_gets_a_single_line(
        self, sim_for, resonant_freq, substrate_thickness_mm
    ):
        params = InsetFedPatchParams(
            resonant_freq=resonant_freq,
            span_freq=resonant_freq / 20,
            substrate_thickness_mm=substrate_thickness_mm,
            substrate_eps_r=3.0,
            substrate_tand=0.001,
            charac_imp=50.0,
            min_trace_spacing_mm=0.05,
        )
        simulation = sim_for(params)
        patch = InsetFedPatchAntenna(params, simulation)
        patch.build_inset_fed_patch_antenna()

        patch.create_mesh()

        copper = params.copper_thickness_mm
        height = params.substrate_thickness_mm
        ground = self.lines_within(simulation, -copper, 0.0)
        trace = self.lines_within(simulation, height, height + copper)
        assert len(ground) == 1
        assert len(trace) == 1
        assert ground[0] == pytest.approx(-copper / 2, abs=1e-6)
        assert trace[0] == pytest.approx(height + copper / 2, abs=1e-6)


@pytest.mark.needs_csxcad
class TestShortingVias:
    def test_each_bandpass_stub_is_grounded_by_cylinders(self, built_bandpass):
        _structure, simulation, params, _ports = built_bandpass
        for stub_number in range(1, params.filter_order + 1):
            vias = primitives_named(simulation.CSX, f"shunt_line_short_{stub_number}")
            assert vias
            for via in vias:
                assert via.GetTypeName() == "Cylinder"
                assert via.GetRadius() == pytest.approx(params.via_diameter_mm / 2)

    @pytest.mark.parametrize(
        ("width_mm", "expected_count"),
        [(0.6, 1), (1.79, 1), (1.8, 2), (3.0, 3), (4.2, 4)],
        ids=["exactly-one", "just-short-of-two", "two", "three", "four"],
    )
    def test_a_wider_stub_gets_more_vias_all_on_the_trace(
        self, bandpass_filter_params, sim_for, width_mm, expected_count
    ):
        params = bandpass_filter_params
        params.via_diameter_mm = 0.6
        params.shunt_line_width_mm = [width_mm] * params.filter_order
        simulation = sim_for(params)
        BandPassQuarterWaveFilter(params, simulation).create_shunt_line_short()

        stub_left = params.line_length_mm
        stub_end = params.series_line_width_mm + params.shunt_line_length_mm[0]
        vias = primitives_named(simulation.CSX, "shunt_line_short_1")
        assert len(vias) == expected_count
        boxes = sorted((bbox(via) for via in vias), key=lambda box: box[0][0])
        for box in boxes:
            assert box[0][0] >= stub_left - 1e-9
            assert box[1][0] <= stub_left + width_mm + 1e-9
            assert box[1][1] <= stub_end + 1e-9
        for left, right in zip(boxes, boxes[1:], strict=False):
            assert right[0][0] - left[1][0] >= params.via_diameter_mm - 1e-9

    def test_a_via_wider_than_a_stub_is_rejected(self, bandpass_filter_params, sim_for):
        bandpass_filter_params.via_diameter_mm = 2 * max(
            bandpass_filter_params.shunt_line_width_mm
        )
        structure = BandPassQuarterWaveFilter(
            bandpass_filter_params, sim_for(bandpass_filter_params)
        )
        with pytest.raises(ValueError, match="via_diameter_mm"):
            structure.create_shunt_line_short()

    def test_the_ifa_short_is_a_cylinder_over_the_ground(self, fr4, sim_for):
        params = InvertedFAntennaParams(resonant_freq=2.45e9, span_freq=1e9, **fr4)
        simulation = sim_for(params)
        InvertedFAntenna(params, simulation).create_short_via()
        (via,) = primitives_named(simulation.CSX, "short_via")
        assert via.GetTypeName() == "Cylinder"
        assert via.GetStart()[1] < 0  # sits over the ground half (y < 0)
