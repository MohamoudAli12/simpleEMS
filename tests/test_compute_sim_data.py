"""Tests for :meth:`SimTools.compute_sim_data` on the FDTD path.

A real run needs the openEMS binary, so these stand stub ports in for
``LumpedPort``: each carries fixed incident/reflected waves and a no-op
``CalcPort``. That isolates what ``compute_sim_data`` itself does -- which
S-parameters it returns for 1 to 4 ports, and what it divides by.
"""

from types import SimpleNamespace

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import SimTools  # noqa: E402


FREQS = np.linspace(1e9, 3e9, 5)


class StubPort:
    """A ``LumpedPort`` stand-in with fixed voltage and current waves."""

    def __init__(self, reflected: complex, incident: complex = 2.0) -> None:
        self.uf_inc = np.full(FREQS.size, incident, dtype=complex)
        self.uf_ref = np.full(FREQS.size, reflected, dtype=complex)
        self.uf_tot = np.full(FREQS.size, 50.0, dtype=complex)
        self.if_tot = np.full(FREQS.size, 1.0, dtype=complex)
        self.calc_port_calls = 0

    def CalcPort(self, *args, **kwargs) -> None:
        self.calc_port_calls += 1


@pytest.fixture
def sim():
    """The three ``SimSetup`` fields the FDTD branch reads."""
    return SimpleNamespace(freqs=FREQS, charac_imp=50.0, backend_engine="FDTD")


def make_ports(count):
    """Port 1 reflects 0.2; port n (n >= 2) sends out 0.1 * n."""
    return [StubPort(0.2)] + [StubPort(0.1 * number) for number in range(2, count + 1)]


class TestComputeSimData:
    @pytest.mark.parametrize(
        ("port_count", "expected_present"),
        [
            (1, []),
            (2, ["s21"]),
            (3, ["s21", "s31"]),
            (4, ["s21", "s31", "s41"]),
        ],
        ids=["one-port", "two-port", "three-port", "four-port"],
    )
    def test_returns_one_transmission_term_per_extra_port(
        self, sim, port_count, expected_present
    ):
        data = SimTools.compute_sim_data(sim, make_ports(port_count))

        for name in ["s21", "s31", "s41"]:
            if name in expected_present:
                assert getattr(data, name) is not None, name
            else:
                assert getattr(data, name) is None, name

    def test_transmission_is_normalised_to_the_driven_incident_wave(self, sim):
        data = SimTools.compute_sim_data(sim, make_ports(4))

        assert data.s11 == pytest.approx(np.full(FREQS.size, 0.2 / 2.0))
        assert data.s21 == pytest.approx(np.full(FREQS.size, 0.2 / 2.0))
        assert data.s31 == pytest.approx(np.full(FREQS.size, 0.3 / 2.0))
        assert data.s41 == pytest.approx(np.full(FREQS.size, 0.4 / 2.0))

    def test_every_port_is_calculated(self, sim):
        ports = make_ports(4)

        SimTools.compute_sim_data(sim, ports)

        assert [port.calc_port_calls for port in ports] == [1, 1, 1, 1]

    def test_bare_port_matches_a_single_element_list(self, sim):
        bare = SimTools.compute_sim_data(sim, StubPort(0.2))
        listed = SimTools.compute_sim_data(sim, [StubPort(0.2)])

        assert bare.s11 == pytest.approx(listed.s11)
        assert bare.z11 == pytest.approx(listed.z11)
        assert bare.s21 is None and listed.s21 is None

    def test_driven_port_sets_impedance_and_power(self, sim):
        data = SimTools.compute_sim_data(sim, make_ports(4))

        assert data.z11 == pytest.approx(np.full(FREQS.size, 50.0))
        assert data.input_power == pytest.approx(np.full(FREQS.size, 25.0))
        assert data.ref_impedance == 50.0

    def test_more_than_four_ports_raises(self, sim):
        with pytest.raises(ValueError, match="up to 4 ports"):
            SimTools.compute_sim_data(sim, make_ports(5))
