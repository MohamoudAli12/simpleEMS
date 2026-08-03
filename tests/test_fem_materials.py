"""Tests for :mod:`simpleEMS.fem_materials`.

The region-ID integers here are described in the module as "the sole contract
between the Gmsh mesh and the GetDP .pro": ``fem_geometry`` tags entities with
them and ``fem_formulation`` references the same numbers. A collision or a
changed constant silently mis-assigns material properties in the solve rather
than raising, so both the values and their disjointness are pinned.
"""

import pytest

from simpleEMS import fem_materials
from simpleEMS.fem_materials import (
    ABC,
    AIR,
    LOSSY_CONDUCTOR,
    PEC,
    PML,
    SYM,
    Dielectric,
    dielectric_region,
    guess_role,
    port_region,
)


# ---------------------------------------------------------------------
# Dielectric
# ---------------------------------------------------------------------
class TestDielectric:
    def test_defaults_describe_vacuum(self):
        material = Dielectric()

        assert material.eps_r == 1.0
        assert material.tan_d == 0.0
        assert material.mu_r == 1.0

    def test_lossless_permittivity_is_real(self):
        assert Dielectric(eps_r=4.4).eps_complex() == 4.4 + 0j

    def test_loss_enters_as_a_negative_imaginary_part(self):
        """Engineering sign convention: ``eps_r * (1 - j*tan_d)``."""
        material = Dielectric(eps_r=4.4, tan_d=0.02)

        assert material.eps_complex() == pytest.approx(4.4 - 0.088j)

    def test_imaginary_part_scales_with_both_eps_and_tand(self):
        assert Dielectric(eps_r=10.0, tan_d=0.01).eps_complex().imag == pytest.approx(
            -0.1
        )

    def test_loss_tangent_is_recoverable_from_the_complex_value(self):
        material = Dielectric(eps_r=3.55, tan_d=0.0027)
        value = material.eps_complex()

        assert -value.imag / value.real == pytest.approx(0.0027, rel=1e-12)


# ---------------------------------------------------------------------
# guess_role
# ---------------------------------------------------------------------
class TestGuessRole:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("substrate", "dielectric"),
            ("dielectric", "dielectric"),
            ("diel_1", "dielectric"),
            ("ground", "pec"),
            ("gnd_plane", "pec"),
            ("patch", "pec"),
            ("trace", "pec"),
            ("feed", "pec"),
            ("microstrip_line", "pec"),
            ("metal", "pec"),
            ("conductor", "pec"),
            ("port_1", "port"),
            ("port", "port"),
        ],
    )
    def test_recognised_names(self, name, expected):
        assert guess_role(name) == expected

    @pytest.mark.parametrize(
        "name", ["SUBSTRATE", "Ground", "PATCH", "Port_1", "FeedLine"]
    )
    def test_matching_is_case_insensitive(self, name):
        assert guess_role(name) is not None

    @pytest.mark.parametrize(
        "name", ["port_feed_1", "feed_port", "port_trace", "port_ground"]
    )
    def test_port_wins_over_conductor_hints(self, name):
        """Documented ordering constraint: a name containing both "port" and a
        conductor hint must classify as a port. Getting this backwards makes
        the excitation vanish into a PEC block, and the solve still runs."""
        assert guess_role(name) == "port"

    @pytest.mark.parametrize("name", ["substrate_metal", "diel_trace"])
    def test_dielectric_wins_over_conductor_hints(self, name):
        """ "substrate"/"diel" precede the PEC hints in the table."""
        assert guess_role(name) == "dielectric"

    @pytest.mark.parametrize("name", ["", "solid1", "Body", "thing", "xyz"])
    def test_unrecognised_names_return_none(self, name):
        assert guess_role(name) is None

    def test_first_matching_hint_wins(self):
        """The lookup returns on the first hit, so table order is behaviour."""
        hints = fem_materials._NAME_ROLE_HINTS

        assert hints[0][0] == "port"
        assert [key for key, _role in hints].index("port") < [
            key for key, _role in hints
        ].index("feed")

    def test_every_hint_maps_to_a_known_role(self):
        roles = {role for _key, role in fem_materials._NAME_ROLE_HINTS}

        assert roles <= {"dielectric", "pec", "port"}

    def test_every_hint_actually_matches_itself(self):
        """A typo'd key would be dead code that never fires."""
        for key, _role in fem_materials._NAME_ROLE_HINTS:
            assert guess_role(key) is not None, f"hint {key!r} never matches"
            assert guess_role(f"my_{key}_solid") is not None


# ---------------------------------------------------------------------
# Region tags
# ---------------------------------------------------------------------
class TestRegionTags:
    def test_fixed_constants(self):
        """These integers appear literally in the generated .pro file; a
        change here must be made in lockstep with fem_formulation."""
        assert AIR == 200
        assert PML == 210
        assert PEC == 300
        assert LOSSY_CONDUCTOR == 350
        assert ABC == 500
        assert SYM == 600

    def test_dielectric_regions_start_at_the_base(self):
        assert dielectric_region(0) == 100
        assert dielectric_region(1) == 101
        assert dielectric_region(7) == 107

    def test_port_regions_are_one_based(self):
        """Port numbering matches the S-parameter numbering, so port 1 -- not
        port 0 -- is the first."""
        assert port_region(1) == 401
        assert port_region(2) == 402

    def test_region_tags_are_injective(self):
        dielectrics = [dielectric_region(i) for i in range(20)]
        ports = [port_region(n) for n in range(1, 21)]

        assert len(set(dielectrics)) == len(dielectrics)
        assert len(set(ports)) == len(ports)

    def test_no_collision_between_region_families(self):
        """The fixed constants must not land inside a generated range for any
        realistic solid or port count."""
        fixed = {AIR, PML, PEC, LOSSY_CONDUCTOR, ABC, SYM}
        dielectrics = {dielectric_region(i) for i in range(50)}
        ports = {port_region(n) for n in range(1, 51)}

        assert fixed & dielectrics == set()
        assert fixed & ports == set()
        assert dielectrics & ports == set()

    def test_dielectric_range_is_bounded_by_the_air_tag(self):
        """Dielectric tags run upward from 100 and AIR is 200, so the scheme
        supports at most 100 dielectrics before it starts overwriting air."""
        assert dielectric_region(99) < AIR
        assert dielectric_region(100) == AIR

    def test_port_range_is_bounded_by_the_abc_tag(self):
        assert port_region(99) < ABC
        assert port_region(100) == ABC
