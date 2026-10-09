"""Tests for the branch-line hybrid coupler (``simpleEMS.coupler``).

The params tests pin the Pozar 7.5 design: Z0 / sqrt(2) series arms, Z0 shunt
arms, and quarter-wave lengths. The geometry tests build the CSXCAD structure
in-process and check the layout of Pozar Figure 7.21; no solver runs. The mesh
contract lives in ``test_geometry.py::TestMeshAcrossStructures``.
"""

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.calc import (  # noqa: E402
    calculate_electrical_length_mm,
    microstrip_width_from_impedance,
)
from simpleEMS.coupler import (  # noqa: E402
    QuadratureBranchLineHybridCoupler,
    QuadratureBranchLineHybridCouplerParams,
)


def bbox(primitive) -> np.ndarray:
    """Return a primitive's bounding box as ``[[xmin, ymin, zmin], [xmax, ...]]``."""
    return np.array(primitive.GetBoundBox(), dtype=float)


def primitives_named(csx, name):
    """All primitives belonging to the property called ``name``."""
    return [p for p in csx.GetAllPrimitives() if p.GetProperty().GetName() == name]


def make_params(fr4, **overrides):
    return QuadratureBranchLineHybridCouplerParams(
        min_freq=1.5e9, max_freq=3.5e9, centre_freq=2.45e9, **{**fr4, **overrides}
    )


# ---------------------------------------------------------------------
# QuadratureBranchLineHybridCouplerParams
# ---------------------------------------------------------------------
class TestCouplerParams:
    def test_frequency_range_is_min_to_max(self, coupler_params):
        assert coupler_params.freq_range == (1.5e9, 3.5e9)

    def test_main_freq_is_the_centre_frequency(self, coupler_params):
        assert coupler_params.main_freq == 2.45e9

    def test_series_arms_are_the_z0_over_root_two_width(self, coupler_params):
        width_mm, _er_eff = microstrip_width_from_impedance(
            50 / np.sqrt(2), 1.6, coupler_params.copper_thickness_mm, 4.4, 2.45e9
        )

        assert coupler_params.series_arm_width_mm == pytest.approx(width_mm, abs=1e-3)

    def test_series_arms_are_wider_than_the_shunt_arms(self, coupler_params):
        """Lower impedance means a wider microstrip."""
        assert coupler_params.series_arm_width_mm > coupler_params.shunt_arm_width_mm

    def test_shunt_arms_and_feed_lines_share_the_z0_width(self, coupler_params):
        assert coupler_params.shunt_arm_width_mm == pytest.approx(
            coupler_params.feed_line_width_mm
        )

    def test_arm_lengths_are_quarter_wave(self, coupler_params):
        _width_mm, er_eff = microstrip_width_from_impedance(
            50, 1.6, coupler_params.copper_thickness_mm, 4.4, 2.45e9
        )

        assert coupler_params.shunt_arm_length_mm == pytest.approx(
            calculate_electrical_length_mm(90, er_eff, 2.45e9), abs=1e-3
        )

    def test_feed_lines_are_an_eighth_wave_by_default(self, coupler_params):
        assert coupler_params.feed_line_length_mm == pytest.approx(
            coupler_params.shunt_arm_length_mm / 2, abs=1e-3
        )

    def test_longer_shunt_arms_follow_the_electrical_length(self, fr4):
        """Pozar suggests lengthening the shunt arms to absorb the junctions."""
        default = make_params(fr4)

        longer = make_params(fr4, shunt_arm_elec_length_deg=100)

        assert longer.shunt_arm_length_mm == pytest.approx(
            default.shunt_arm_length_mm * 100 / 90, abs=1e-2
        )

    def test_substrate_length_spans_the_arms_and_both_feeds(self, coupler_params):
        expected = (
            coupler_params.series_arm_length_mm
            + coupler_params.shunt_arm_width_mm
            + 2 * coupler_params.feed_line_length_mm
        )

        assert coupler_params.substrate_length_mm == pytest.approx(expected)

    def test_substrate_width_pads_by_six_thicknesses_a_side(self, coupler_params):
        expected = (
            coupler_params.shunt_arm_length_mm
            + coupler_params.series_arm_width_mm
            + 12 * 1.6
        )

        assert coupler_params.substrate_width_mm == pytest.approx(expected)

    def test_default_simulation_box_adds_a_wavelength(self, coupler_params):
        box = coupler_params._default_simulation_box

        assert box == pytest.approx(
            [
                coupler_params.substrate_length_mm + coupler_params.lambda0,
                coupler_params.substrate_width_mm + coupler_params.lambda0,
                2 * coupler_params.lambda0,
            ]
        )

    def test_geometry_is_rounded_to_fp_precision(self, coupler_params):
        for name in [
            "series_arm_width_mm",
            "series_arm_length_mm",
            "shunt_arm_width_mm",
            "shunt_arm_length_mm",
            "feed_line_width_mm",
            "feed_line_length_mm",
        ]:
            value = getattr(coupler_params, name)
            assert value == np.round(value, coupler_params.fp_precision)

    def test_too_narrow_a_trace_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Series arm width"):
            make_params(fr4, min_trace_width_mm=10.0)

    def test_too_small_a_gap_between_the_series_arms_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Gap between series arms"):
            make_params(fr4, min_trace_spacing_mm=50.0)


# ---------------------------------------------------------------------
# QuadratureBranchLineHybridCoupler geometry
# ---------------------------------------------------------------------
class TestCouplerGeometry:
    def test_builds_the_expected_properties(self, built_coupler):
        _structure, sim, _params, _ports = built_coupler

        names = {prop.GetName() for prop in sim.CSX.GetAllProperties()}

        assert {"substrate", "ground", "branch_arms", "feed_lines"} <= names

    def test_substrate_is_centred_and_matches_the_params(self, built_coupler):
        _structure, sim, params, _ports = built_coupler

        (substrate,) = primitives_named(sim.CSX, "substrate")

        assert bbox(substrate).ravel() == pytest.approx(
            [
                -params.substrate_length_mm / 2,
                -params.substrate_width_mm / 2,
                0,
                params.substrate_length_mm / 2,
                params.substrate_width_mm / 2,
                params.substrate_thickness_mm,
            ]
        )

    def test_ground_sits_beneath_the_whole_substrate(self, built_coupler):
        _structure, sim, params, _ports = built_coupler

        (ground,) = primitives_named(sim.CSX, "ground")
        limits = bbox(ground)

        assert limits[0][2] == pytest.approx(-params.copper_thickness_mm)
        assert limits[1][2] == pytest.approx(0.0)
        assert limits[1][0] - limits[0][0] == pytest.approx(params.substrate_length_mm)
        assert limits[1][1] - limits[0][1] == pytest.approx(params.substrate_width_mm)

    def test_four_branch_arms(self, built_coupler):
        _structure, sim, _params, _ports = built_coupler

        assert len(primitives_named(sim.CSX, "branch_arms")) == 4

    def test_series_arms_run_along_x_at_the_shunt_arm_ends(self, built_coupler):
        _structure, sim, params, _ports = built_coupler

        arms = [bbox(arm) for arm in primitives_named(sim.CSX, "branch_arms")]
        series_arms = [
            arm for arm in arms if arm[1][0] - arm[0][0] > params.series_arm_length_mm
        ]
        centres_y = sorted((arm[0][1] + arm[1][1]) / 2 for arm in series_arms)

        assert centres_y == pytest.approx(
            [-params.shunt_arm_length_mm / 2, params.shunt_arm_length_mm / 2]
        )
        for arm in series_arms:
            assert arm[1][1] - arm[0][1] == pytest.approx(params.series_arm_width_mm)
            assert arm[1][0] - arm[0][0] == pytest.approx(
                params.series_arm_length_mm + params.shunt_arm_width_mm
            )

    def test_shunt_arms_fill_the_gap_between_the_series_arms(self, built_coupler):
        """The shunt arms must touch both series arms, or the hybrid is open."""
        _structure, sim, params, _ports = built_coupler

        arms = [bbox(arm) for arm in primitives_named(sim.CSX, "branch_arms")]
        shunt_arms = [
            arm
            for arm in arms
            if arm[1][0] - arm[0][0] == pytest.approx(params.shunt_arm_width_mm)
        ]
        inner_edge_y = (params.shunt_arm_length_mm - params.series_arm_width_mm) / 2

        assert len(shunt_arms) == 2
        for arm in shunt_arms:
            assert arm[0][1] == pytest.approx(-inner_edge_y)
            assert arm[1][1] == pytest.approx(inner_edge_y)
        centres_x = sorted((arm[0][0] + arm[1][0]) / 2 for arm in shunt_arms)
        assert centres_x == pytest.approx(
            [-params.series_arm_length_mm / 2, params.series_arm_length_mm / 2]
        )

    def test_four_feed_lines_reach_the_board_edges(self, built_coupler):
        _structure, sim, params, _ports = built_coupler

        feeds = [bbox(feed) for feed in primitives_named(sim.CSX, "feed_lines")]

        assert len(feeds) == 4
        for feed in feeds:
            outer_x = max(abs(feed[0][0]), abs(feed[1][0]))
            assert outer_x == pytest.approx(params.substrate_length_mm / 2)
            assert feed[1][1] - feed[0][1] == pytest.approx(params.feed_line_width_mm)

    def test_all_metal_is_coplanar_on_the_substrate(self, built_coupler):
        _structure, sim, params, _ports = built_coupler

        for name in ("branch_arms", "feed_lines"):
            for primitive in primitives_named(sim.CSX, name):
                limits = bbox(primitive)
                assert limits[0][2] == pytest.approx(params.substrate_thickness_mm)
                assert limits[1][2] == pytest.approx(
                    params.substrate_thickness_mm + params.copper_thickness_mm
                )

    def test_returns_four_ports_with_only_the_input_excited(self, built_coupler):
        _structure, _sim, _params, ports = built_coupler

        assert [port.number for port in ports] == [1, 2, 3, 4]
        assert [port.excite for port in ports] == [1, 0, 0, 0]

    def test_ports_follow_pozar_figure_7_21(self, built_coupler):
        """Input top left, through top right, coupled bottom right, isolated
        bottom left."""
        _structure, _sim, params, ports = built_coupler

        half_x = params.substrate_length_mm / 2
        half_y = params.shunt_arm_length_mm / 2
        expected = [
            (-half_x, half_y),
            (half_x, half_y),
            (half_x, -half_y),
            (-half_x, -half_y),
        ]

        for port, (position_x, position_y) in zip(ports, expected, strict=True):
            assert port.start[0] == pytest.approx(position_x)
            assert (port.start[1] + port.stop[1]) / 2 == pytest.approx(position_y)

    def test_ports_bridge_ground_to_trace(self, built_coupler):
        _structure, _sim, params, ports = built_coupler

        for port in ports:
            assert port.start[2] == pytest.approx(0.0)
            assert port.stop[2] == pytest.approx(
                params.substrate_thickness_mm + params.copper_thickness_mm
            )

    def test_manual_mesh_honours_a_user_simulation_box(self, coupler_params, sim_for):
        params = dataclasses.replace(coupler_params, simulation_box=[100, 100, 60])
        sim = sim_for(params)
        structure = QuadratureBranchLineHybridCoupler(params, sim)
        structure.build_quadrature_branch_line_hybrid_coupler()

        structure.create_mesh(manual_mesh=True)

        grid = sim.CSX.GetGrid()
        assert grid.GetLines(0)[0] == pytest.approx(-50)
        assert grid.GetLines(0)[-1] == pytest.approx(50)
