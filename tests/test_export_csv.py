"""Tests for ``SimTools.export_csv``.

The CSVs go into Excel or another plotting tool, so the assertions read the
file back as plain comma-separated numbers and check the header and the values
a user would plot.
"""

from types import SimpleNamespace

import numpy as np
import pytest

# simpleEMS imports CSXCAD/openEMS at module scope, so without them this
# module cannot even be collected. Skip cleanly rather than erroring.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.sim_tools import EfficiencyCurve, NF2FFData, SimTools  # noqa: E402


FREQS = np.linspace(1e9, 3e9, 21)
S11 = 0.3 * np.exp(1j * FREQS / 1e9)
S21 = 0.9 * np.ones_like(FREQS) + 0j


def read_header(path):
    return path.read_text().splitlines()[0].split(",")


def read_values(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


class TestOnePort:
    @pytest.fixture
    def exported(self, tmp_path):
        (path,) = SimTools.export_csv(FREQS, S11, output_path=tmp_path)
        return path

    def test_writes_into_a_csv_subdirectory(self, exported, tmp_path):
        assert exported == tmp_path / "csv" / "s_param.csv"
        assert exported.is_file()

    def test_header_names_every_column(self, exported):
        assert read_header(exported) == [
            "Frequency (GHz)",
            "|S11| (dB)",
            "S11 phase (deg)",
            "Re(S11)",
            "Im(S11)",
        ]

    def test_frequency_is_in_ghz(self, exported):
        assert read_values(exported)[:, 0] == pytest.approx(FREQS / 1e9)

    def test_values_match_the_input(self, exported):
        values = read_values(exported)

        assert values[:, 1] == pytest.approx(20 * np.log10(np.abs(S11)))
        assert values[:, 2] == pytest.approx(np.angle(S11, deg=True))
        assert values[:, 3] + 1j * values[:, 4] == pytest.approx(S11)

    def test_prefix_goes_in_front_of_the_file_name(self, tmp_path):
        (path,) = SimTools.export_csv(FREQS, S11, output_path=tmp_path, prefix="patch_")

        assert path == tmp_path / "csv" / "patch_s_param.csv"

    def test_defaults_to_sim_path_under_cwd(self, tmp_path):
        # _no_stray_writes has already chdir'd into tmp_path.
        (path,) = SimTools.export_csv(FREQS, S11)

        assert path == tmp_path / "Sim_Path" / "csv" / "s_param.csv"


class TestTwoPort:
    def test_adds_s21_columns(self, tmp_path):
        (path,) = SimTools.export_csv(FREQS, S11, s21=S21, output_path=tmp_path)
        values = read_values(path)

        assert read_header(path)[5:] == [
            "|S21| (dB)",
            "S21 phase (deg)",
            "Re(S21)",
            "Im(S21)",
        ]
        assert values[:, 7] + 1j * values[:, 8] == pytest.approx(S21)


class TestSMatrix:
    def test_writes_every_entry_in_row_order(self, tmp_path):
        matrix = np.zeros((FREQS.size, 3, 3), dtype=complex)
        matrix[:, 1, 0] = S21
        (path,) = SimTools.export_csv(FREQS, s_matrix=matrix, output_path=tmp_path)

        names = [column for column in read_header(path) if column.startswith("Re(")]
        assert names == [
            f"Re(S{row}{column})" for row in range(1, 4) for column in range(1, 4)
        ]
        # S21 is the fourth entry; its real part sits in column 1 + 3*4 + 2.
        assert read_values(path)[:, 15] == pytest.approx(S21.real)

    def test_a_zero_entry_writes_a_finite_db_value(self, tmp_path):
        matrix = np.zeros((FREQS.size, 2, 2), dtype=complex)
        (path,) = SimTools.export_csv(FREQS, s_matrix=matrix, output_path=tmp_path)

        assert np.isfinite(read_values(path)).all()


class TestInvalidInput:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"freqs": FREQS},
            {"freqs": FREQS, "s11": S11, "s_matrix": np.zeros((FREQS.size, 1, 1))},
            {"freqs": FREQS, "s11": S11[:-1]},
            {"s11": S11},
        ],
        ids=["nothing", "freqs-alone", "s11-and-matrix", "length-mismatch", "no-freqs"],
    )
    def test_raises_value_error(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            SimTools.export_csv(output_path=tmp_path, **kwargs)


# ---------------------------------------------------------------------
# Far field
# ---------------------------------------------------------------------
ANGLES = np.linspace(-180.0, 180.0, 3601)
DMAX = 4.0
EFFICIENCY = 0.5
MISMATCH = 0.8


def make_cut(e_norm):
    """An ``nf2ff_results`` stand-in carrying what the export reads."""
    return SimpleNamespace(
        theta=np.deg2rad(ANGLES), E_norm=[e_norm[:, np.newaxis]], Dmax=[DMAX]
    )


@pytest.fixture
def nf2ff_data():
    # The xz cut peaks at the sphere's peak; the yz cut sits 6 dB below it
    # and the xy cut has an exact null at its first sample.
    xz_field = np.cos(np.deg2rad(ANGLES) / 2) ** 2
    yz_field = 0.5 * np.ones_like(ANGLES)
    xy_field = np.ones_like(ANGLES)
    xy_field[0] = 0.0
    sphere = SimpleNamespace(E_norm=[np.ones((91, 181))], Dmax=[DMAX])
    curve_freqs = np.linspace(2.4e9, 2.5e9, 11)
    return NF2FFData(
        freq=2.45e9,
        sphere=sphere,
        xz_cut=make_cut(xz_field),
        yz_cut=make_cut(yz_field),
        xy_cut=make_cut(xy_field),
        efficiency=EFFICIENCY,
        mismatch_efficiency=MISMATCH,
        efficiency_curve=EfficiencyCurve(
            freqs=curve_freqs,
            radiation=np.full(11, 0.9),
            total=np.linspace(0.5, 0.9, 11),
            computed_freqs=curve_freqs[::5],
            computed_radiation=np.full(3, 0.9),
            computed_total=np.array([0.5, 0.7, 0.9]),
        ),
    )


class TestFarField:
    @pytest.fixture
    def exported(self, nf2ff_data, tmp_path):
        pattern, efficiency = SimTools.export_csv(
            nf2ff_data=nf2ff_data, output_path=tmp_path
        )
        return pattern, efficiency

    def test_writes_a_separate_file_for_pattern_and_efficiency(
        self, exported, tmp_path
    ):
        assert exported == (
            tmp_path / "csv" / "radiation_pattern.csv",
            tmp_path / "csv" / "efficiency.csv",
        )

    def test_pattern_header_names_every_plane(self, exported):
        assert read_header(exported[0]) == [
            "Angle (deg)",
            "xz-plane directivity (dBi)",
            "xz-plane realized gain (dBi)",
            "yz-plane directivity (dBi)",
            "yz-plane realized gain (dBi)",
            "xy-plane directivity (dBi)",
            "xy-plane realized gain (dBi)",
        ]

    def test_pattern_angles_are_in_degrees(self, exported):
        assert read_values(exported[0])[:, 0] == pytest.approx(ANGLES)

    def test_peak_directivity_is_dmax(self, exported):
        peak = read_values(exported[0])[:, 1].max()

        assert peak == pytest.approx(10 * np.log10(DMAX))

    def test_a_cut_below_the_peak_sits_below_dmax(self, exported):
        yz_directivity = read_values(exported[0])[:, 3]

        assert yz_directivity == pytest.approx(10 * np.log10(DMAX) - 20 * np.log10(2))

    def test_gain_is_directivity_less_both_efficiencies(self, exported):
        values = read_values(exported[0])

        offset = 10 * np.log10(EFFICIENCY * MISMATCH)
        assert values[:, 2] == pytest.approx(values[:, 1] + offset)

    def test_a_null_writes_finite_values(self, exported):
        assert np.isfinite(read_values(exported[0])).all()

    def test_efficiency_header(self, exported):
        assert read_header(exported[1]) == [
            "Frequency (GHz)",
            "Radiation efficiency (%)",
            "Total efficiency (%)",
            "Radiation efficiency (dB)",
            "Total efficiency (dB)",
        ]

    def test_efficiency_values_match_the_curve(self, exported, nf2ff_data):
        values = read_values(exported[1])
        curve = nf2ff_data.efficiency_curve

        assert values[:, 0] == pytest.approx(curve.freqs / 1e9)
        assert values[:, 2] == pytest.approx(100 * curve.total)
        assert values[:, 4] == pytest.approx(10 * np.log10(curve.total))


class TestEverythingInOneCall:
    def test_writes_all_three_files(self, nf2ff_data, tmp_path):
        paths = SimTools.export_csv(
            FREQS, S11, nf2ff_data=nf2ff_data, output_path=tmp_path, prefix="patch_"
        )

        assert [path.name for path in paths] == [
            "patch_s_param.csv",
            "patch_radiation_pattern.csv",
            "patch_efficiency.csv",
        ]
