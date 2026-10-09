"""Property tests for :mod:`simpleEMS.calc`.

Every params class sizes its geometry through these formulas, so they must
behave across the whole range of boards people actually build, not just at
the FR-4 / 2.45 GHz point the example tests pin. The strategies below span
that envelope; outside it the closed forms are not meant to hold.
"""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from openEMS.physical_constants import C0  # noqa: E402

from simpleEMS.calc import (  # noqa: E402
    branch_line_hybrid_impedances,
    calculate_electrical_length_mm,
    dipole_resonant_length_mm,
    microstrip_impedance,
    microstrip_width_from_impedance,
    patch_dims,
    ungrounded_strip_eps_eff,
)

COPPER_THICKNESS_MM = 0.035

impedances = st.floats(min_value=25.0, max_value=110.0)
widths_mm = st.floats(min_value=0.1, max_value=10.0)
heights_mm = st.floats(min_value=0.2, max_value=3.2)
permittivities = st.floats(min_value=2.0, max_value=10.2)
frequencies = st.floats(min_value=0.5e9, max_value=10e9)
eps_effs = st.floats(min_value=1.0, max_value=10.0)
angles_deg = st.floats(min_value=1.0, max_value=360.0)


class TestMicrostripImpedance:
    @given(widths_mm, heights_mm, permittivities, frequencies)
    def test_eps_eff_lies_between_air_and_the_substrate(
        self, width, height, eps_r, frequency
    ):
        _, eps_eff = microstrip_impedance(
            width, height, COPPER_THICKNESS_MM, eps_r, frequency
        )

        assert 1.0 < eps_eff < eps_r

    @given(
        st.lists(widths_mm, min_size=2, max_size=2, unique=True),
        heights_mm,
        permittivities,
        frequencies,
    )
    def test_a_wider_trace_has_a_lower_impedance(
        self, two_widths, height, eps_r, frequency
    ):
        narrow, wide = sorted(two_widths)
        assume(wide - narrow > 1e-3)

        narrow_imp, _ = microstrip_impedance(
            narrow, height, COPPER_THICKNESS_MM, eps_r, frequency
        )
        wide_imp, _ = microstrip_impedance(
            wide, height, COPPER_THICKNESS_MM, eps_r, frequency
        )

        assert wide_imp < narrow_imp

    @given(impedances, heights_mm, permittivities, frequencies)
    def test_the_synthesised_width_gives_back_the_target_impedance(
        self, target_imp, height, eps_r, frequency
    ):
        width, eps_eff = microstrip_width_from_impedance(
            target_imp, height, COPPER_THICKNESS_MM, eps_r, frequency
        )
        impedance, check_eps_eff = microstrip_impedance(
            width, height, COPPER_THICKNESS_MM, eps_r, frequency
        )

        assert impedance == pytest.approx(target_imp, abs=0.05)
        assert eps_eff == check_eps_eff


class TestElectricalLength:
    @given(angles_deg, eps_effs, frequencies)
    def test_degrees_and_radians_agree(self, angle, eps_eff, frequency):
        from_degrees = calculate_electrical_length_mm(angle, eps_eff, frequency)
        from_radians = calculate_electrical_length_mm(
            np.deg2rad(angle), eps_eff, frequency, radians=True
        )

        assert from_degrees == pytest.approx(from_radians, rel=1e-12)

    @given(angles_deg, eps_effs, frequencies)
    def test_length_is_the_angle_fraction_of_a_guided_wavelength(
        self, angle, eps_eff, frequency
    ):
        guided_wavelength_mm = C0 / (frequency * np.sqrt(eps_eff)) * 1e3

        assert calculate_electrical_length_mm(
            angle, eps_eff, frequency
        ) == pytest.approx(angle / 360.0 * guided_wavelength_mm, rel=1e-12)


class TestBranchLineHybrid:
    @given(impedances)
    def test_series_arms_are_the_shunt_arms_over_root_two(self, port_imp):
        series_imp, shunt_imp = branch_line_hybrid_impedances(port_imp)

        assert shunt_imp == port_imp
        assert series_imp * np.sqrt(2) == pytest.approx(shunt_imp, rel=1e-12)


class TestPrintedAntennaHelpers:
    @given(frequencies, eps_effs)
    def test_dipole_shrinks_with_the_guided_wavelength(self, frequency, eps_eff):
        in_air = dipole_resonant_length_mm(frequency)
        on_board = dipole_resonant_length_mm(frequency, eps_eff)

        assert on_board == pytest.approx(in_air / np.sqrt(eps_eff), rel=1e-12)

    @given(widths_mm, heights_mm, permittivities, frequencies)
    def test_ungrounded_eps_eff_lies_between_air_and_the_half_space(
        self, width, height, eps_r, frequency
    ):
        eps_eff = ungrounded_strip_eps_eff(width, height, eps_r, frequency)

        assert 1.0 <= eps_eff <= (eps_r + 1) / 2

    @given(
        widths_mm,
        st.lists(heights_mm, min_size=2, max_size=2, unique=True),
        permittivities,
        frequencies,
    )
    def test_a_thicker_substrate_raises_the_ungrounded_eps_eff(
        self, width, two_heights, eps_r, frequency
    ):
        thin, thick = sorted(two_heights)

        assert ungrounded_strip_eps_eff(
            width, thin, eps_r, frequency
        ) <= ungrounded_strip_eps_eff(width, thick, eps_r, frequency)


class TestPatchDims:
    @given(
        frequency=st.floats(min_value=1e9, max_value=10e9),
        eps_r=permittivities,
        height_mm=st.floats(min_value=0.5, max_value=1.6),
        port_imp=st.floats(min_value=40.0, max_value=75.0),
    )
    def test_inset_fits_inside_half_the_patch(
        self, frequency, eps_r, height_mm, port_imp
    ):
        dims = patch_dims(frequency, eps_r, height_mm * 1e-3, port_imp, 0.035)

        assert dims.patch_width_mm > 0
        assert dims.patch_length_mm > 0
        assert 0 <= dims.inset_length_mm <= dims.patch_length_mm / 2
        assert dims.probe_pos_mm == dims.inset_length_mm
        assert 0 < dims.inset_width_mm < dims.patch_width_mm
