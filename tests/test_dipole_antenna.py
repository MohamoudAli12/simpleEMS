"""Tests for the printed half-wave dipole (``simpleEMS.dipole_antenna``).

The params tests pin the Zingg design: the arm width and substrate set
``eps_eff``, which sets the guided wavelength the dipole is cut to. The
geometry tests build the CSXCAD structure in-process; no solver runs. The
mesh contract lives in ``test_geometry.py::TestMeshAcrossStructures``.
"""

import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.calc import (  # noqa: E402
    dipole_resonant_length_mm,
    ungrounded_strip_eps_eff,
)
from simpleEMS.dipole_antenna import (  # noqa: E402
    PrintedDipoleAntenna,
    PrintedDipoleAntennaParams,
)

C0 = 299792458.0


def bbox(primitive) -> np.ndarray:
    """Return a primitive's bounding box as ``[[xmin, ymin, zmin], [xmax, ...]]``."""
    return np.array(primitive.GetBoundBox(), dtype=float)


def primitives_named(csx, name):
    """All primitives belonging to the property called ``name``."""
    return [p for p in csx.GetAllPrimitives() if p.GetProperty().GetName() == name]


def make_params(fr4, **overrides):
    return PrintedDipoleAntennaParams(
        min_freq=1.5e9, max_freq=3.5e9, resonant_freq=2.45e9, **{**fr4, **overrides}
    )


# ---------------------------------------------------------------------
# PrintedDipoleAntennaParams
# ---------------------------------------------------------------------
class TestDipoleParams:
    def test_frequency_range_is_min_to_max(self, dipole_params):
        assert dipole_params.freq_range == (1.5e9, 3.5e9)

    def test_main_freq_is_the_resonant_frequency(self, dipole_params):
        assert dipole_params.main_freq == 2.45e9

    def test_eps_eff_comes_from_the_ungrounded_strip_model(self, dipole_params):
        expected = ungrounded_strip_eps_eff(1.0, 1.6, 4.4, 2.45e9)

        assert dipole_params.eps_eff == pytest.approx(expected, abs=1e-3)

    def test_eps_eff_lies_between_air_and_the_substrate(self, dipole_params):
        assert 1.0 < dipole_params.eps_eff < 4.4

    def test_guided_wavelength_is_shortened_by_eps_eff(self, dipole_params):
        expected = 1e3 * C0 / (2.45e9 * np.sqrt(dipole_params.eps_eff))

        assert dipole_params.guided_wavelength_mm == pytest.approx(expected, abs=1e-2)

    def test_dipole_length_is_047_guided_wavelengths(self, dipole_params):
        expected = dipole_resonant_length_mm(2.45e9, dipole_params.eps_eff, 0.47)

        assert dipole_params.dipole_length_mm == pytest.approx(expected, abs=1e-2)

    def test_arms_and_gap_add_up_to_the_dipole_length(self, dipole_params):
        total = 2 * dipole_params.arm_length_mm + dipole_params.feed_gap_mm

        assert total == pytest.approx(dipole_params.dipole_length_mm, abs=2e-3)

    def test_feed_gap_scales_with_the_guided_wavelength(self, dipole_params):
        assert dipole_params.feed_gap_mm == pytest.approx(
            0.01 * dipole_params.guided_wavelength_mm, abs=1e-3
        )

    def test_substrate_pads_by_six_thicknesses_a_side(self, dipole_params):
        assert dipole_params.substrate_length_mm == pytest.approx(
            dipole_params.dipole_length_mm + 12 * 1.6
        )
        assert dipole_params.substrate_width_mm == pytest.approx(1.0 + 12 * 1.6)

    def test_default_simulation_box_adds_a_wavelength(self, dipole_params):
        box = dipole_params._default_simulation_box

        assert box == pytest.approx(
            [
                dipole_params.substrate_length_mm + dipole_params.lambda0,
                dipole_params.substrate_width_mm + dipole_params.lambda0,
                2 * dipole_params.lambda0,
            ]
        )

    def test_the_arm_width_sets_eps_eff(self, fr4):
        params = make_params(fr4, arm_width_mm=5.0)

        expected = ungrounded_strip_eps_eff(5.0, 1.6, 4.4, 2.45e9)
        assert params.eps_eff == pytest.approx(expected, abs=1e-3)

    def test_geometry_is_rounded_to_fp_precision(self, dipole_params):
        for name in [
            "eps_eff",
            "guided_wavelength_mm",
            "dipole_length_mm",
            "arm_length_mm",
            "feed_gap_mm",
        ]:
            value = getattr(dipole_params, name)
            assert value == np.round(value, dipole_params.fp_precision)

    def test_too_narrow_an_arm_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Arm width"):
            make_params(fr4, arm_width_mm=0.05)

    def test_too_small_a_feed_gap_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="Feed gap"):
            make_params(fr4, feed_gap_wavelengths=1e-4)

    def test_a_feed_gap_longer_than_the_dipole_is_rejected(self, fr4):
        with pytest.raises(ValueError, match="leaves no room for the arms"):
            make_params(fr4, feed_gap_wavelengths=0.5)


# ---------------------------------------------------------------------
# PrintedDipoleAntenna geometry
# ---------------------------------------------------------------------
class TestDipoleGeometry:
    def test_builds_a_substrate_and_arms_but_no_ground(self, built_dipole):
        _antenna, sim, _params, _port = built_dipole

        names = {prop.GetName() for prop in sim.CSX.GetAllProperties()}

        assert {"substrate", "dipole_arms"} <= names
        assert "ground" not in names

    def test_substrate_is_centred_on_the_origin(self, built_dipole):
        _antenna, sim, params, _port = built_dipole

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

    def test_two_arms_run_from_the_gap_to_the_tips(self, built_dipole):
        _antenna, sim, params, _port = built_dipole

        arms = sorted(
            (bbox(arm) for arm in primitives_named(sim.CSX, "dipole_arms")),
            key=lambda limits: limits[0][0],
        )

        assert len(arms) == 2
        assert arms[0][0][0] == pytest.approx(-params.dipole_length_mm / 2)
        assert arms[0][1][0] == pytest.approx(-params.feed_gap_mm / 2)
        assert arms[1][0][0] == pytest.approx(params.feed_gap_mm / 2)
        assert arms[1][1][0] == pytest.approx(params.dipole_length_mm / 2)

    def test_arms_are_centred_in_y_and_sit_on_the_substrate(self, built_dipole):
        _antenna, sim, params, _port = built_dipole

        for arm in primitives_named(sim.CSX, "dipole_arms"):
            limits = bbox(arm)
            assert limits[0][1] == pytest.approx(-params.arm_width_mm / 2)
            assert limits[1][1] == pytest.approx(params.arm_width_mm / 2)
            assert limits[0][2] == pytest.approx(params.substrate_thickness_mm)

    def test_port_is_excited_and_drives_across_the_gap(self, built_dipole):
        _antenna, _sim, params, port = built_dipole

        assert port.number == 1
        assert port.excite == 1
        assert port.start[0] == pytest.approx(-params.feed_gap_mm / 2)
        assert port.stop[0] == pytest.approx(params.feed_gap_mm / 2)

    def test_port_uses_the_declared_impedance(self, built_dipole):
        _antenna, _sim, _params, port = built_dipole

        resistance = port.R
        assert resistance == pytest.approx(73)

    def test_manual_mesh_honours_a_user_simulation_box(self, dipole_params, sim_for):
        params = dataclasses.replace(dipole_params, simulation_box=[200, 100, 80])
        sim = sim_for(params)
        antenna = PrintedDipoleAntenna(params, sim)
        antenna.build_printed_dipole_antenna()

        antenna.create_mesh(manual_mesh=True)

        x_lines = sim.CSX.GetGrid().GetLines(0)
        assert x_lines[0] == pytest.approx(-100)
        assert x_lines[-1] == pytest.approx(100)

    def test_manual_mesh_grids_the_feed_gap(self, built_dipole):
        antenna, sim, params, _port = built_dipole

        antenna.create_mesh(manual_mesh=True)

        x_lines = np.asarray(sim.CSX.GetGrid().GetLines(0))
        in_gap = x_lines[np.abs(x_lines) <= params.feed_gap_mm / 2 + 1e-9]
        assert len(in_gap) >= 3
