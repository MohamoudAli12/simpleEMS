"""Tests for ``SimTools.export_touchstone``.

Touchstone is the interchange format users take into ADS, Qucs, or scikit-rf,
so the assertions here read the file back with :mod:`skrf` rather than
inspecting text: what matters is that another tool recovers the same numbers.
"""

from pathlib import Path

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

import skrf  # noqa: E402

from simpleEMS.sim_tools import SimTools  # noqa: E402


FREQS = np.linspace(1e9, 3e9, 21)
S11 = 0.3 * np.exp(1j * FREQS / 1e9)
S21 = 0.9 * np.ones_like(FREQS) + 0j


def read(path: Path) -> skrf.Network:
    return skrf.Network(str(path))


# ---------------------------------------------------------------------
# One-port
# ---------------------------------------------------------------------
class TestOnePort:
    @pytest.fixture
    def exported(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, output_path=tmp_path)
        return tmp_path / "touchstone" / "s_param.s1p"

    def test_writes_an_s1p_into_a_touchstone_subdirectory(self, exported):
        assert exported.is_file()

    def test_creates_the_directory_if_missing(self, tmp_path):
        target = tmp_path / "nested" / "deeper"

        SimTools.export_touchstone(FREQS, S11, output_path=target)

        assert (target / "touchstone" / "s_param.s1p").is_file()

    def test_reports_one_port(self, exported):
        assert read(exported).nports == 1

    def test_frequencies_round_trip(self, exported):
        network = read(exported)

        assert network.f == pytest.approx(FREQS, rel=1e-9)

    def test_s11_round_trips(self, exported):
        network = read(exported)

        assert network.s[:, 0, 0] == pytest.approx(S11, abs=1e-9)

    def test_default_reference_impedance_is_50_ohm(self, exported):
        assert read(exported).z0[0, 0] == pytest.approx(50.0)

    def test_custom_reference_impedance_is_recorded(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, charac_imp=75.0, output_path=tmp_path)

        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert network.z0[0, 0] == pytest.approx(75.0)

    def test_custom_filename_is_used(self, tmp_path):
        SimTools.export_touchstone(
            FREQS, S11, output_path=tmp_path, filename="patch_s11"
        )

        assert (tmp_path / "touchstone" / "patch_s11.s1p").is_file()

    def test_default_output_path_is_sim_path_under_cwd(self, tmp_path):
        """Documented default. The autouse cwd fixture keeps this out of the
        repository."""
        SimTools.export_touchstone(FREQS, S11)

        assert (Path.cwd() / "Sim_Path" / "touchstone" / "s_param.s1p").is_file()


# ---------------------------------------------------------------------
# Two-port
# ---------------------------------------------------------------------
class TestTwoPort:
    @pytest.fixture
    def exported(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, s21=S21, output_path=tmp_path)
        return tmp_path / "touchstone" / "s_param.s2p"

    def test_passing_s21_selects_the_s2p_extension(self, exported):
        assert exported.is_file()
        assert exported.suffix == ".s2p"

    def test_reports_two_ports(self, exported):
        assert read(exported).nports == 2

    def test_s11_and_s21_round_trip(self, exported):
        network = read(exported)

        assert network.s[:, 0, 0] == pytest.approx(S11, abs=1e-9)
        assert network.s[:, 1, 0] == pytest.approx(S21, abs=1e-9)

    def test_reverse_parameters_are_written_as_zero(self, exported):
        """Documented limitation: only the forward direction is simulated, so
        S12 and S22 are placeholders. Pinning it here means a future change to
        populate them is a deliberate one."""
        network = read(exported)

        assert network.s[:, 0, 1] == pytest.approx(np.zeros(len(FREQS)), abs=1e-12)
        assert network.s[:, 1, 1] == pytest.approx(np.zeros(len(FREQS)), abs=1e-12)

    def test_frequencies_round_trip(self, exported):
        assert read(exported).f == pytest.approx(FREQS, rel=1e-9)

    def test_reference_impedance_applies_to_both_ports(self, tmp_path):
        SimTools.export_touchstone(
            FREQS, S11, s21=S21, charac_imp=75.0, output_path=tmp_path
        )

        network = read(tmp_path / "touchstone" / "s_param.s2p")

        assert network.z0[0] == pytest.approx([75.0, 75.0])


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------
class TestRobustness:
    def test_explicit_none_s21_takes_the_one_port_path(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, s21=None, output_path=tmp_path)

        assert (tmp_path / "touchstone" / "s_param.s1p").is_file()
        assert not (tmp_path / "touchstone" / "s_param.s2p").exists()

    def test_s21_is_keyword_only(self):
        """Guards the signature: a positional S21 would land on ``freqs``."""
        with pytest.raises(TypeError):
            SimTools.export_touchstone(FREQS, S11, S21)

    def test_single_frequency_point(self, tmp_path):
        SimTools.export_touchstone(
            np.array([2.45e9]), np.array([0.1 + 0.2j]), output_path=tmp_path
        )

        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert network.nports == 1
        assert len(network.f) == 1

    def test_repeated_export_overwrites_cleanly(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, output_path=tmp_path)
        SimTools.export_touchstone(FREQS, S11 * 0.5, output_path=tmp_path)

        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert network.s[:, 0, 0] == pytest.approx(S11 * 0.5, abs=1e-9)

    def test_lossless_reflection_survives_the_round_trip(self, tmp_path):
        """|S11| = 1 is the passivity boundary and a common numerical edge."""
        unity = np.ones_like(FREQS) + 0j

        SimTools.export_touchstone(FREQS, unity, output_path=tmp_path)
        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert np.abs(network.s[:, 0, 0]) == pytest.approx(
            np.ones(len(FREQS)), abs=1e-9
        )
