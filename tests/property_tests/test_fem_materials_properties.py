"""Property tests for :mod:`simpleEMS.fem_materials`.

The region tags are the only contract between the Gmsh mesh and the GetDP
problem file, so two regions sharing a tag would silently merge them. These
tests check that no dielectric or port tag can collide with another tag.
"""

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

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

FIXED_REGIONS = {AIR, PML, PEC, LOSSY_CONDUCTOR, ABC, SYM}

# Solid names as CAD tools write them; full Unicode case folding is out of scope
solid_names = st.text(alphabet=string.ascii_letters + string.digits + "_-", max_size=30)
# Realistic counts: a board has a handful of dielectrics and ports
dielectric_indices = st.integers(min_value=0, max_value=50)
port_numbers = st.integers(min_value=1, max_value=50)


class TestRegionTags:
    @given(index=dielectric_indices, number=port_numbers)
    def test_dielectric_and_port_tags_never_collide(self, index, number):
        assert dielectric_region(index) != port_region(number)

    @given(index=dielectric_indices)
    def test_dielectric_tags_avoid_the_fixed_regions(self, index):
        assert dielectric_region(index) not in FIXED_REGIONS

    @given(number=port_numbers)
    def test_port_tags_avoid_the_fixed_regions(self, number):
        assert port_region(number) not in FIXED_REGIONS

    @given(first=dielectric_indices, second=dielectric_indices)
    def test_distinct_dielectrics_get_distinct_tags(self, first, second):
        assert (dielectric_region(first) == dielectric_region(second)) == (
            first == second
        )

    @given(first=port_numbers, second=port_numbers)
    def test_distinct_ports_get_distinct_tags(self, first, second):
        assert (port_region(first) == port_region(second)) == (first == second)


class TestDielectric:
    @given(
        eps_r=st.floats(min_value=1.0, max_value=100.0),
        tan_d=st.floats(min_value=0.0, max_value=0.1),
    )
    def test_complex_permittivity_keeps_eps_r_and_is_lossy(self, eps_r, tan_d):
        permittivity = Dielectric(eps_r=eps_r, tan_d=tan_d).eps_complex()

        assert permittivity.real == pytest.approx(eps_r)
        assert permittivity.imag <= 0.0
        assert -permittivity.imag == pytest.approx(eps_r * tan_d)


class TestGuessRole:
    @given(name=solid_names)
    def test_role_ignores_letter_case(self, name):
        assert guess_role(name.upper()) == guess_role(name.lower())

    @given(prefix=st.text(max_size=10), suffix=st.text(max_size=10))
    def test_a_port_in_the_name_always_wins(self, prefix, suffix):
        assert guess_role(f"{prefix}port{suffix}") == "port"
