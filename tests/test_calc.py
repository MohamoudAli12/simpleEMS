"""Unit tests for :mod:`simpleEMS.calc` -- the closed-form design formulas.

These are the numerical foundation the whole library sits on: every params
class calls into this module, so a silent regression here changes every
generated geometry. Reference values are taken from Balanis (3rd ed.) and the
Hammerstad-Jensen papers cited in the module's own docstrings.
"""

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.calc import (  # noqa: E402
    PatchDims,
    calculate_electrical_length_mm,
    conductance_G1,
    inset_depth,
    microstrip_impedance,
    microstrip_width_from_impedance,
    patch_dims,
)


C0 = 299792458.0


# ---------------------------------------------------------------------
# calculate_electrical_length_mm
# ---------------------------------------------------------------------
class TestElectricalLength:
    def test_quarter_wave_is_quarter_of_guided_wavelength(self):
        """A 90-degree line is lambda_g / 4 by definition."""
        eps_eff, freq = 3.3, 2.45e9
        lambda_g_mm = 1e3 * C0 / (freq * np.sqrt(eps_eff))

        length = calculate_electrical_length_mm(90, eps_eff, freq)

        assert length == pytest.approx(lambda_g_mm / 4, rel=1e-12)

    def test_half_wave_is_twice_quarter_wave(self):
        quarter = calculate_electrical_length_mm(90, 3.3, 2.45e9)
        half = calculate_electrical_length_mm(180, 3.3, 2.45e9)

        assert half == pytest.approx(2 * quarter, rel=1e-12)

    def test_radians_flag_matches_equivalent_degrees(self):
        deg = calculate_electrical_length_mm(90, 3.3, 2.45e9)
        rad = calculate_electrical_length_mm(np.pi / 2, 3.3, 2.45e9, radians=True)

        assert rad == pytest.approx(deg, rel=1e-12)

    def test_length_scales_inversely_with_frequency(self):
        low = calculate_electrical_length_mm(90, 3.3, 1e9)
        high = calculate_electrical_length_mm(90, 3.3, 2e9)

        assert low == pytest.approx(2 * high, rel=1e-12)

    def test_zero_phase_gives_zero_length(self):
        assert calculate_electrical_length_mm(0, 3.3, 2.45e9) == 0.0


# ---------------------------------------------------------------------
# microstrip_impedance
# ---------------------------------------------------------------------
class TestMicrostripImpedance:
    def test_known_50_ohm_line_on_fr4(self):
        """~3.0 mm on 1.6 mm FR-4 is the textbook 50 ohm geometry."""
        z, eps_eff = microstrip_impedance(3.0, 1.6, 0.035, 4.4, 2.45e9)

        assert z == pytest.approx(50.0, abs=1.0)
        assert eps_eff == pytest.approx(3.36, abs=0.1)

    def test_eps_eff_lies_between_one_and_eps_r(self):
        """A partly air-filled line cannot be slower than the bulk dielectric."""
        for width in (0.1, 0.5, 1.0, 3.0, 10.0, 30.0):
            _, eps_eff = microstrip_impedance(width, 1.6, 0.035, 4.4, 2.45e9)
            assert 1.0 < eps_eff < 4.4

    def test_impedance_decreases_with_width(self):
        widths = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        zs = [microstrip_impedance(w, 1.6, 0.035, 4.4, 2.45e9)[0] for w in widths]

        assert zs == sorted(zs, reverse=True)

    def test_impedance_decreases_with_permittivity(self):
        eps_values = [2.2, 3.5, 4.4, 6.15, 10.2]
        zs = [microstrip_impedance(3.0, 1.6, 0.035, er, 2.45e9)[0] for er in eps_values]

        assert zs == sorted(zs, reverse=True)

    @pytest.mark.parametrize("f_hz", [0, -1.0])
    def test_non_positive_frequency_returns_static_solution(self, f_hz):
        """``f_hz <= 0`` is the documented escape hatch for the dispersion-free
        result, so both branches must agree."""
        zero = microstrip_impedance(3.0, 1.6, 0.035, 4.4, f_hz)
        negative = microstrip_impedance(3.0, 1.6, 0.035, 4.4, -1.0)

        assert zero == negative

    def test_dispersion_raises_eps_eff_above_static(self):
        """Microstrip dispersion pulls eps_eff toward eps_r as frequency rises."""
        _, static = microstrip_impedance(3.0, 1.6, 0.035, 4.4, 0)
        _, at_2g = microstrip_impedance(3.0, 1.6, 0.035, 4.4, 2.45e9)
        _, at_20g = microstrip_impedance(3.0, 1.6, 0.035, 4.4, 20e9)

        assert static < at_2g < at_20g < 4.4

    def test_zero_thickness_is_handled(self):
        """``t == 0`` is substituted with 1e-6 rather than dividing by zero."""
        z_zero, _ = microstrip_impedance(3.0, 1.6, 0.0, 4.4, 2.45e9)
        z_thin, _ = microstrip_impedance(3.0, 1.6, 1e-6, 4.4, 2.45e9)

        assert np.isfinite(z_zero)
        assert z_zero == pytest.approx(z_thin, rel=1e-9)

    def test_thicker_copper_lowers_impedance(self):
        """Thickness correction widens the effective trace."""
        thin, _ = microstrip_impedance(3.0, 1.6, 0.018, 4.4, 2.45e9)
        thick, _ = microstrip_impedance(3.0, 1.6, 0.070, 4.4, 2.45e9)

        assert thick < thin


# ---------------------------------------------------------------------
# microstrip_width_from_impedance
# ---------------------------------------------------------------------
class TestMicrostripWidthFromImpedance:
    @pytest.mark.parametrize("eps_r", [2.2, 3.5, 4.4, 6.15, 10.2])
    @pytest.mark.parametrize("height", [0.254, 0.8, 1.6, 3.2])
    @pytest.mark.parametrize("target_z", [25, 50, 75, 100])
    @pytest.mark.parametrize("freq", [0, 2.45e9, 28e9])
    def test_round_trip_converges_within_tolerance(self, eps_r, height, target_z, freq):
        """width -> Z round-trip must land inside the search tolerance.

        The binary search runs a fixed 100 iterations and returns whatever it
        holds without asserting convergence, so this is the only thing
        standing between a non-converged width and a silently wrong geometry.
        """
        try:
            width, _ = microstrip_width_from_impedance(
                target_z, height, 0.035, eps_r, freq, tolerance=0.01
            )
        except ValueError:
            pytest.skip(f"{target_z} ohm not realizable on this substrate")

        achieved, _ = microstrip_impedance(width, height, 0.035, eps_r, freq)

        assert achieved == pytest.approx(target_z, abs=0.01)

    def test_known_50_ohm_width_on_fr4(self):
        width, eps_eff = microstrip_width_from_impedance(50, 1.6, 0.035, 4.4, 2.45e9)

        assert width == pytest.approx(3.06, abs=0.15)
        assert eps_eff == pytest.approx(3.37, abs=0.15)

    def test_returned_eps_eff_matches_returned_width(self):
        """The second return value must describe the *same* line as the first."""
        width, eps_eff = microstrip_width_from_impedance(50, 1.6, 0.035, 4.4, 2.45e9)
        _, direct = microstrip_impedance(width, 1.6, 0.035, 4.4, 2.45e9)

        assert eps_eff == pytest.approx(direct, rel=1e-12)

    def test_higher_impedance_gives_narrower_trace(self):
        narrow, _ = microstrip_width_from_impedance(75, 1.6, 0.035, 4.4, 2.45e9)
        wide, _ = microstrip_width_from_impedance(25, 1.6, 0.035, 4.4, 2.45e9)

        assert narrow < wide

    def test_width_stays_inside_search_bracket(self):
        width, _ = microstrip_width_from_impedance(50, 1.6, 0.035, 4.4, 2.45e9)

        assert 0.001 <= width <= 100

    @pytest.mark.parametrize("unreachable_z", [1.0, 5000.0])
    def test_unrealizable_impedance_raises(self, unreachable_z):
        with pytest.raises(ValueError, match="not realizable"):
            microstrip_width_from_impedance(unreachable_z, 1.6, 0.035, 4.4, 2.45e9)

    def test_error_message_reports_the_achievable_range(self):
        with pytest.raises(ValueError) as excinfo:
            microstrip_width_from_impedance(5000.0, 1.6, 0.035, 4.4, 2.45e9)

        assert "range" in str(excinfo.value)

    def test_looser_tolerance_still_lands_near_target(self):
        width, _ = microstrip_width_from_impedance(
            50, 1.6, 0.035, 4.4, 2.45e9, tolerance=1.0
        )
        achieved, _ = microstrip_impedance(width, 1.6, 0.035, 4.4, 2.45e9)

        assert achieved == pytest.approx(50, abs=1.0)


# ---------------------------------------------------------------------
# conductance_G1
# ---------------------------------------------------------------------
class TestConductanceG1:
    def test_matches_balanis_narrow_slot_approximation(self):
        """For W << lambda0, Balanis 14-8 gives G1 ~= (1/90)(W/lambda0)^2."""
        freq = 2.45e9
        lambda0 = C0 / freq
        width = lambda0 / 20  # comfortably in the narrow-slot regime

        g1 = conductance_G1(width, freq)
        approx = (1 / 90) * (width / lambda0) ** 2

        assert g1 == pytest.approx(approx, rel=0.05)

    def test_typical_patch_conductance_is_of_expected_order(self):
        """A 2.45 GHz FR-4 patch slot sits around 1 mS."""
        g1 = conductance_G1(0.0375, 2.45e9)

        assert 1e-4 < g1 < 1e-2

    def test_conductance_is_positive_across_widths(self):
        for width in (0.001, 0.01, 0.0375, 0.1, 0.5):
            assert conductance_G1(width, 2.45e9) > 0

    def test_conductance_grows_with_slot_width(self):
        widths = [0.005, 0.01, 0.02, 0.0375]
        values = [conductance_G1(w, 2.45e9) for w in widths]

        assert values == sorted(values)

    def test_depends_only_on_electrical_width(self):
        """G1 is a function of W/lambda0, so doubling both leaves it unchanged."""
        a = conductance_G1(0.0375, 2.45e9)
        b = conductance_G1(0.0750, 1.225e9)

        assert a == pytest.approx(b, rel=1e-6)

    def test_integrand_singularity_guard_does_not_break_result(self):
        """The integrand zeroes itself near cos(theta) == 0; the integral must
        still be finite and smooth across that region."""
        values = [conductance_G1(w, 2.45e9) for w in np.linspace(0.03, 0.045, 15)]

        assert all(np.isfinite(values))
        assert values == sorted(values)


# ---------------------------------------------------------------------
# inset_depth
# ---------------------------------------------------------------------
class TestInsetDepth:
    def test_inset_and_probe_position_are_identical(self):
        inset, probe = inset_depth(50, 0.0375, 0.028, 2.45e9)

        assert inset == probe

    def test_feeding_at_edge_resistance_needs_no_inset(self):
        """At R_edge the arccos argument is 1, so the depth collapses to zero."""
        width, length, freq = 0.0375, 0.028, 2.45e9
        r_edge = 1.0 / (2.0 * conductance_G1(width, freq))

        inset, _ = inset_depth(r_edge, width, length, freq)

        assert inset == pytest.approx(0.0, abs=1e-12)

    def test_depth_is_half_the_patch_at_zero_impedance(self):
        """arccos(0) = pi/2, so a short at the feed sits at the patch centre."""
        length = 0.028
        inset, _ = inset_depth(0.0, 0.0375, length, 2.45e9)

        assert inset == pytest.approx(length / 2, rel=1e-12)

    def test_lower_impedance_needs_deeper_inset(self):
        depths = [inset_depth(z, 0.0375, 0.028, 2.45e9)[0] for z in (100, 75, 50, 25)]

        assert depths == sorted(depths)

    def test_inset_stays_within_the_patch(self):
        length = 0.028
        for z in (10, 25, 50, 100, 150):
            inset, _ = inset_depth(z, 0.0375, length, 2.45e9)
            assert 0 <= inset <= length / 2

    def test_impedance_above_edge_resistance_raises(self):
        """An inset feed transforms the edge resistance down, never up."""
        width, length, freq = 0.0375, 0.028, 2.45e9
        r_edge = 1.0 / (2.0 * conductance_G1(width, freq))

        with pytest.raises(ValueError, match="edge resistance"):
            inset_depth(r_edge * 1.5, width, length, freq)


# ---------------------------------------------------------------------
# patch_dims
# ---------------------------------------------------------------------
class TestPatchDims:
    def test_returns_named_tuple_with_documented_fields(self):
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)

        assert isinstance(dims, PatchDims)
        assert dims._fields == (
            "patch_width_mm",
            "patch_length_mm",
            "inset_length_mm",
            "inset_width_mm",
            "probe_pos_mm",
        )

    def test_matches_published_2450_mhz_fr4_patch(self):
        """The canonical worked example: 2.45 GHz, eps_r 4.4, h 1.6 mm."""
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)

        assert dims.patch_width_mm == pytest.approx(37.2, abs=0.5)
        assert dims.patch_length_mm == pytest.approx(28.8, abs=0.5)

    def test_patch_is_wider_than_it_is_long(self):
        """W uses sqrt(2/(er+1)); L uses the shorter guided half-wavelength."""
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)

        assert dims.patch_width_mm > dims.patch_length_mm

    def test_width_follows_the_closed_form(self):
        freq, eps_r = 2.45e9, 4.4
        expected_mm = 1e3 * C0 / (2 * freq) * np.sqrt(2 / (eps_r + 1))

        dims = patch_dims(freq, eps_r, 1.6e-3, 50, 0.035)

        assert dims.patch_width_mm == pytest.approx(expected_mm, rel=1e-12)

    def test_dimensions_scale_inversely_with_frequency(self):
        low = patch_dims(1.0e9, 4.4, 1.6e-3, 50, 0.035)
        high = patch_dims(2.0e9, 4.4, 1.6e-3, 50, 0.035)

        assert low.patch_width_mm == pytest.approx(2 * high.patch_width_mm, rel=1e-12)
        # length also carries the fringing correction, so only approximately
        assert low.patch_length_mm == pytest.approx(2 * high.patch_length_mm, rel=0.05)

    def test_higher_permittivity_shrinks_the_patch(self):
        low_er = patch_dims(2.45e9, 2.2, 1.6e-3, 50, 0.035)
        high_er = patch_dims(2.45e9, 10.2, 1.6e-3, 50, 0.035)

        assert high_er.patch_width_mm < low_er.patch_width_mm
        assert high_er.patch_length_mm < low_er.patch_length_mm

    def test_inset_length_agrees_with_inset_depth(self):
        """``patch_dims`` must not diverge from the function it delegates to."""
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)
        direct, _ = inset_depth(
            50, dims.patch_width_mm / 1e3, dims.patch_length_mm / 1e3, 2.45e9
        )

        assert dims.inset_length_mm == pytest.approx(direct * 1e3, rel=1e-9)

    def test_inset_width_agrees_with_feed_line_width(self):
        """The inset slot is sized from the feed line's 50 ohm width."""
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)
        direct, _ = microstrip_width_from_impedance(50, 1.6, 0.035, 4.4, 2.45e9)

        assert dims.inset_width_mm == pytest.approx(direct, rel=1e-9)

    def test_probe_position_mirrors_inset_length(self):
        dims = patch_dims(2.45e9, 4.4, 1.6e-3, 50, 0.035)

        assert dims.probe_pos_mm == pytest.approx(dims.inset_length_mm, rel=1e-12)

    def test_all_dimensions_positive(self):
        dims = patch_dims(5.8e9, 3.5, 0.8e-3, 50, 0.035)

        assert all(value > 0 for value in dims)

    def test_unreachable_impedance_propagates_value_error(self):
        with pytest.raises(ValueError):
            patch_dims(2.45e9, 4.4, 1.6e-3, 5000, 0.035)
