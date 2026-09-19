"""Tests for ``SimTools.export_touchstone``.

Touchstone is the interchange format users take into ADS, Qucs, or scikit-rf,
so the assertions here read the file back with ``read_touchstone`` rather than
inspecting text: what matters is that a reader recovers the same numbers. The
byte-level layout is pinned separately in ``test_touchstone.py``.
"""

from pathlib import Path

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.export_touchstone import TouchstoneData, read_touchstone  # noqa: E402
from simpleEMS.sim_tools import SimTools  # noqa: E402


FREQS = np.linspace(1e9, 3e9, 21)
S11 = 0.3 * np.exp(1j * FREQS / 1e9)
S21 = 0.9 * np.ones_like(FREQS) + 0j


def read(path: Path) -> TouchstoneData:
    return read_touchstone(path)


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
        assert read(exported).s_matrix.shape[1] == 1

    def test_frequencies_round_trip(self, exported):
        network = read(exported)

        assert network.freqs == pytest.approx(FREQS, rel=1e-9)

    def test_s11_round_trips(self, exported):
        network = read(exported)

        assert network.s_matrix[:, 0, 0] == pytest.approx(S11, abs=1e-9)

    def test_default_reference_impedance_is_50_ohm(self, exported):
        assert read(exported).ref_impedance == pytest.approx(50.0)

    def test_custom_reference_impedance_is_recorded(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, charac_imp=75.0, output_path=tmp_path)

        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert network.ref_impedance == pytest.approx(75.0)

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
        assert read(exported).s_matrix.shape[1] == 2

    def test_s11_and_s21_round_trip(self, exported):
        network = read(exported)

        assert network.s_matrix[:, 0, 0] == pytest.approx(S11, abs=1e-9)
        assert network.s_matrix[:, 1, 0] == pytest.approx(S21, abs=1e-9)

    def test_reverse_parameters_are_written_as_zero(self, exported):
        """Documented limitation: only the forward direction is simulated, so
        S12 and S22 are placeholders. Pinning it here means a future change to
        populate them is a deliberate one."""
        network = read(exported)

        assert network.s_matrix[:, 0, 1] == pytest.approx(
            np.zeros(len(FREQS)), abs=1e-12
        )
        assert network.s_matrix[:, 1, 1] == pytest.approx(
            np.zeros(len(FREQS)), abs=1e-12
        )

    def test_frequencies_round_trip(self, exported):
        assert read(exported).freqs == pytest.approx(FREQS, rel=1e-9)

    def test_reference_impedance_applies_to_both_ports(self, tmp_path):
        SimTools.export_touchstone(
            FREQS, S11, s21=S21, charac_imp=75.0, output_path=tmp_path
        )

        network = read(tmp_path / "touchstone" / "s_param.s2p")

        assert network.ref_impedance == pytest.approx(75.0)


# ---------------------------------------------------------------------
# Full S-matrix
# ---------------------------------------------------------------------
class TestSMatrix:
    @pytest.fixture
    def full_matrix(self):
        matrix = np.empty((len(FREQS), 2, 2), dtype=complex)
        matrix[:, 0, 0] = S11
        matrix[:, 1, 0] = S21
        matrix[:, 0, 1] = 0.8 * S21
        matrix[:, 1, 1] = -S11
        return matrix

    def test_every_parameter_round_trips(self, tmp_path, full_matrix):
        path = SimTools.export_touchstone(
            FREQS, s_matrix=full_matrix, output_path=tmp_path
        )

        assert path == tmp_path / "touchstone" / "s_param.s2p"
        assert read(path).s_matrix == pytest.approx(full_matrix, abs=1e-12)

    def test_port_count_sets_the_extension(self, tmp_path):
        matrix = np.ones((len(FREQS), 3, 3), dtype=complex)

        path = SimTools.export_touchstone(FREQS, s_matrix=matrix, output_path=tmp_path)

        assert path.suffix == ".s3p"

    @pytest.mark.parametrize(
        "extra",
        [{"s11": S11}, {"s21": S21}],
        ids=["with-s11", "with-s21"],
    )
    def test_s_matrix_rejects_s11_or_s21(self, tmp_path, full_matrix, extra):
        with pytest.raises(ValueError, match="not both"):
            SimTools.export_touchstone(
                FREQS, s_matrix=full_matrix, output_path=tmp_path, **extra
            )

    def test_neither_s11_nor_s_matrix_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="either"):
            SimTools.export_touchstone(FREQS, output_path=tmp_path)


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

        assert network.s_matrix.shape[1] == 1
        assert len(network.freqs) == 1

    def test_repeated_export_overwrites_cleanly(self, tmp_path):
        SimTools.export_touchstone(FREQS, S11, output_path=tmp_path)
        SimTools.export_touchstone(FREQS, S11 * 0.5, output_path=tmp_path)

        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert network.s_matrix[:, 0, 0] == pytest.approx(S11 * 0.5, abs=1e-9)

    def test_lossless_reflection_survives_the_round_trip(self, tmp_path):
        """|S11| = 1 is the passivity boundary and a common numerical edge."""
        unity = np.ones_like(FREQS) + 0j

        SimTools.export_touchstone(FREQS, unity, output_path=tmp_path)
        network = read(tmp_path / "touchstone" / "s_param.s1p")

        assert np.abs(network.s_matrix[:, 0, 0]) == pytest.approx(
            np.ones(len(FREQS)), abs=1e-9
        )
