"""Tests for the printed quarter-wave monopole (``simpleEMS.monopole_antenna``).

The params tests pin the design: a ``Z0`` microstrip feed over a partial
ground plane, and a radiating strip half the length of the equivalent printed
dipole. The geometry tests build the CSXCAD structure in-process; no solver
runs. The mesh contract lives in
``test_geometry.py::TestMeshAcrossStructures``.
"""

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.calc import (  # noqa: E402
    dipole_resonant_length_mm,
    microstrip_width_from_impedance,
    ungrounded_strip_eps_eff,
)
from simpleEMS.monopole_antenna import (  # noqa: E402
    PrintedMonopoleAntenna,
    PrintedMonopoleAntennaParams,
)

C0 = 299792458.0
LAMBDA0_MM = 1e3 * C0 / 2.45e9


def bbox(primitive) -> np.ndarray:
    """Return a primitive's bounding box as ``[[xmin, ymin, zmin], [xmax, ...]]``."""
    return np.array(primitive.GetBoundBox(), dtype=float)


def only_primitive(csx, name):
    """The single primitive belonging to the property called ``name``."""
    (primitive,) = [
        p for p in csx.GetAllPrimitives() if p.GetProperty().GetName() == name
    ]
    return primitive


def make_params(fr4, **overrides):
    return PrintedMonopoleAntennaParams(
        min_freq=1.5e9, max_freq=3.5e9, resonant_freq=2.45e9, **{**fr4, **overrides}
    )


# ---------------------------------------------------------------------
# PrintedMonopoleAntennaParams
# ---------------------------------------------------------------------
class TestMonopoleParams:
    def test_frequency_range_is_min_to_max(self, monopole_params):
        assert monopole_params.freq_range == (1.5e9, 3.5e9)

    def test_main_freq_is_the_resonant_frequency(self, monopole_params):
        assert monopole_params.main_freq == 2.45e9

    def test_feed_line_is_the_z0_microstrip_width(self, monopole_params):
        width_mm, _er_eff = microstrip_width_from_impedance(
            50, 1.6, monopole_params.copper_thickness_mm, 4.4, 2.45e9
        )

        assert monopole_params.feed_line_width_mm == pytest.approx(width_mm, abs=1e-3)

    def test_radiator_defaults_to_the_feed_line_width(self, monopole_params):
        assert monopole_params.radiator_width_mm == pytest.approx(
            monopole_params.feed_line_width_mm
        )

    def test_a_given_radiator_width_is_kept(self, fr4):
        params = make_params(fr4, radiator_width_mm=1.5)

        assert params.radiator_width_mm == pytest.approx(1.5)

    def test_eps_eff_follows_the_radiator_width(self, monopole_params):
        expected = ungrounded_strip_eps_eff(
            monopole_params.radiator_width_mm, 1.6, 4.4, 2.45e9
        )

        assert monopole_params.eps_eff == pytest.approx(expected, abs=1e-3)

    def test_guided_wavelength_is_shortened_by_eps_eff(self, monopole_params):
        expected = LAMBDA0_MM / np.sqrt(monopole_params.eps_eff)

        assert monopole_params.guided_wavelength_mm == pytest.approx(expected, abs=1e-2)

    def test_monopole_is_half_the_equivalent_dipole(self, monopole_params):
        dipole_mm = dipole_resonant_length_mm(2.45e9, monopole_params.eps_eff, 0.47)

        assert monopole_params.monopole_length_mm == pytest.approx(
            dipole_mm / 2, abs=1e-2
        )

    def test_ground_scales_with_the_free_space_wavelength(self, monopole_params):
        assert monopole_params.ground_length_mm == pytest.approx(
            0.25 * LAMBDA0_MM, abs=1e-3
        )
        assert monopole_params.ground_width_mm == pytest.approx(
            0.4 * LAMBDA0_MM, abs=1e-3
        )

    def test_substrate_runs_from_the_port_past_the_tip(self, monopole_params):
        expected = (
            monopole_params.ground_length_mm
            + monopole_params.monopole_length_mm
            + 6 * 1.6
        )

        assert monopole_params.substrate_length_mm == pytest.approx(expected)
        assert monopole_params.substrate_width_mm == monopole_params.ground_width_mm

    def test_default_simulation_box_adds_a_wavelength(self, monopole_params):
        box = monopole_params._default_simulation_box

        assert box == pytest.approx(
            [
                monopole_params.substrate_length_mm + monopole_params.lambda0,
                monopole_params.substrate_width_mm + monopole_params.lambda0,
                2 * monopole_params.lambda0,
            ]
        )

    def test_geometry_is_rounded_to_fp_precision(self, monopole_params):
        for name in [
            "feed_line_width_mm",
            "radiator_width_mm",
            "eps_eff",
            "guided_wavelength_mm",
            "monopole_length_mm",
            "ground_length_mm",
            "ground_width_mm",
        ]:
            value = getattr(monopole_params, name)
            assert value == np.round(value, monopole_params.fp_precision)

    def test_too_narrow_a_feed_line_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Feed line width"):
            make_params(fr4, min_trace_width_mm=10.0)

    def test_too_narrow_a_radiator_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Radiator width"):
            make_params(fr4, radiator_width_mm=0.05)

    def test_a_radiator_wider_than_the_board_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="board width"):
            make_params(fr4, radiator_width_mm=60.0)


# ---------------------------------------------------------------------
# PrintedMonopoleAntenna geometry
# ---------------------------------------------------------------------
class TestMonopoleGeometry:
    def test_builds_the_expected_properties(self, built_monopole):
        _antenna, sim, _params, _port = built_monopole

        names = {prop.GetName() for prop in sim.CSX.GetAllProperties()}

        assert {"substrate", "ground", "feed_line", "radiator"} <= names

    def test_substrate_runs_from_the_port_edge(self, built_monopole):
        _antenna, sim, params, _port = built_monopole

        limits = bbox(only_primitive(sim.CSX, "substrate"))

        assert limits[0][0] == pytest.approx(-params.ground_length_mm)
        assert limits[1][0] - limits[0][0] == pytest.approx(params.substrate_length_mm)
        assert limits[1][1] - limits[0][1] == pytest.approx(params.substrate_width_mm)

    def test_ground_stops_at_the_feed_point(self, built_monopole):
        """The board under the radiating strip must stay clear of ground."""
        _antenna, sim, params, _port = built_monopole

        limits = bbox(only_primitive(sim.CSX, "ground"))

        assert limits[0][0] == pytest.approx(-params.ground_length_mm)
        assert limits[1][0] == pytest.approx(0.0)
        assert limits[0][2] == pytest.approx(-params.copper_thickness_mm)

    def test_feed_line_runs_over_the_ground(self, built_monopole):
        _antenna, sim, params, _port = built_monopole

        limits = bbox(only_primitive(sim.CSX, "feed_line"))

        assert limits[0][0] == pytest.approx(-params.ground_length_mm)
        assert limits[1][0] == pytest.approx(0.0)
        assert limits[1][1] - limits[0][1] == pytest.approx(params.feed_line_width_mm)

    def test_radiator_continues_the_feed_to_its_tip(self, built_monopole):
        _antenna, sim, params, _port = built_monopole

        limits = bbox(only_primitive(sim.CSX, "radiator"))

        assert limits[0][0] == pytest.approx(0.0)
        assert limits[1][0] == pytest.approx(params.monopole_length_mm)
        assert limits[1][1] - limits[0][1] == pytest.approx(params.radiator_width_mm)
        assert limits[0][2] == pytest.approx(params.substrate_thickness_mm)

    def test_port_sits_on_the_board_edge_and_bridges_to_ground(self, built_monopole):
        _antenna, _sim, params, port = built_monopole

        assert port.number == 1
        assert port.excite == 1
        assert port.start[0] == pytest.approx(-params.ground_length_mm)
        assert port.start[2] == pytest.approx(0.0)
        assert port.stop[2] == pytest.approx(
            params.substrate_thickness_mm + params.copper_thickness_mm
        )

    def test_manual_mesh_centres_the_default_box_on_the_board(self, built_monopole):
        antenna, sim, params, _port = built_monopole

        antenna.create_mesh(manual_mesh=True)

        x_lines = sim.CSX.GetGrid().GetLines(0)
        board_centre_x = (params.substrate_length_mm - 2 * params.ground_length_mm) / 2
        assert (x_lines[0] + x_lines[-1]) / 2 == pytest.approx(board_centre_x, abs=1e-3)

    def test_manual_mesh_honours_a_user_simulation_box(self, monopole_params, sim_for):
        params = dataclasses.replace(
            monopole_params, simulation_box=[[-80, 80], [-60, 60], [-40, 40]]
        )
        sim = sim_for(params)
        antenna = PrintedMonopoleAntenna(params, sim)
        antenna.build_printed_monopole_antenna()

        antenna.create_mesh(manual_mesh=True)

        x_lines = sim.CSX.GetGrid().GetLines(0)
        assert x_lines[0] == pytest.approx(-80)
        assert x_lines[-1] == pytest.approx(80)
