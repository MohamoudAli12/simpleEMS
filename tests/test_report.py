"""Tests for the PDF design report (``simpleEMS.report``)."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad


def _sim_data(sim, params, synthetic_resonator, *, two_port=False, s11=None):
    """A ``SimData`` built from the analytic resonator instead of a solve."""
    from simpleEMS.sim_tools import SimData

    freqs = sim.freqs
    if s11 is None:
        # The offset keeps |S11| off an exact zero at resonance
        s11 = np.array(
            [synthetic_resonator(freq, f0=params.main_freq)[0, 0] for freq in freqs]
        )
        s11 = 0.98 * s11 + 0.01
    s21 = (
        np.sqrt(1.0 - np.abs(s11) ** 2) * np.exp(-1j * freqs / 1e9)
        if two_port
        else None
    )
    z11 = sim.charac_imp * (1.0 + s11) / (1.0 - s11)
    s11_magnitude = np.clip(np.abs(s11), 0, 0.999)
    vswr = (1.0 + s11_magnitude) / (1.0 - s11_magnitude)
    return SimData(
        freqs, s11, s21, None, None, z11, vswr, 1.0, z11, np.ones_like(z11), 50.0
    )


def _far_field(freq):
    """A hand-made ``NF2FFData`` with a single lobe pointing along +z."""
    from simpleEMS.sim_tools import EfficiencyCurve, NF2FFData

    theta = np.deg2rad(np.arange(0.0, 181.0, 2.0))
    phi = np.deg2rad(np.arange(0.0, 361.0, 2.0))
    sphere_field = np.cos(theta / 2.0)[:, None] * np.ones_like(phi)[None, :]
    sphere = SimpleNamespace(
        theta=theta, phi=phi, E_norm=sphere_field[None], Dmax=np.array([4.0])
    )
    cut_angles = np.deg2rad(np.arange(-180.0, 180.05, 0.1))
    cut = SimpleNamespace(
        theta=cut_angles,
        phi=cut_angles,
        E_norm=np.abs(np.cos(cut_angles / 2.0))[None, :, None],
    )
    curve_freqs = np.linspace(0.98 * freq, 1.02 * freq, 5)
    curve = EfficiencyCurve(
        freqs=curve_freqs,
        radiation=np.full(5, 0.8),
        total=np.full(5, 0.75),
        computed_freqs=curve_freqs,
        computed_radiation=np.full(5, 0.8),
        computed_total=np.full(5, 0.75),
    )
    return NF2FFData(
        freq=freq,
        sphere=sphere,
        xz_cut=cut,
        yz_cut=cut,
        xy_cut=cut,
        efficiency=0.8,
        mismatch_efficiency=0.95,
        efficiency_curve=curve,
    )


def _is_pdf(path):
    return path.is_file() and path.read_bytes().startswith(b"%PDF")


class TestGenerateReport:
    def test_writes_an_antenna_report_with_its_far_field(
        self, built_inset, synthetic_resonator, tmp_path
    ):
        from simpleEMS.report import generate_report

        _antenna, sim, params, _port = built_inset
        sim_data = _sim_data(sim, params, synthetic_resonator)

        report_path = generate_report(
            params,
            sim,
            sim_data,
            _far_field(params.main_freq),
            output_path=tmp_path,
            author="Test author",
        )

        assert report_path == tmp_path / "report" / "design_report.pdf"
        assert _is_pdf(report_path)

    def test_writes_a_two_port_filter_report(
        self, built_bandpass, synthetic_resonator, tmp_path
    ):
        from simpleEMS.report import generate_report

        _structure, sim, params, _ports = built_bandpass
        sim_data = _sim_data(sim, params, synthetic_resonator, two_port=True)

        report_path = generate_report(params, sim, sim_data, output_path=tmp_path)

        assert _is_pdf(report_path)

    def test_writes_a_report_for_a_design_that_never_matches(
        self, built_inset, synthetic_resonator, tmp_path
    ):
        from simpleEMS.report import generate_report

        _antenna, sim, params, _port = built_inset
        unmatched = np.full(len(sim.freqs), 0.7 + 0.0j)
        sim_data = _sim_data(sim, params, synthetic_resonator, s11=unmatched)

        report_path = generate_report(params, sim, sim_data, output_path=tmp_path)

        assert _is_pdf(report_path)

    def test_leaves_no_pyplot_figure_open(
        self, built_inset, synthetic_resonator, tmp_path
    ):
        from simpleEMS.report import generate_report

        _antenna, sim, params, _port = built_inset
        sim_data = _sim_data(sim, params, synthetic_resonator)

        generate_report(params, sim, sim_data, output_path=tmp_path)

        assert plt.get_fignums() == []

    def test_simtools_wrapper_defaults_to_sim_path(
        self, built_inset, synthetic_resonator, tmp_path
    ):
        from simpleEMS.sim_tools import SimTools

        _antenna, sim, params, _port = built_inset
        sim_data = _sim_data(sim, params, synthetic_resonator)

        report_path = SimTools.generate_report(params, sim, sim_data)

        assert report_path == tmp_path / "Sim_Path" / "report" / "design_report.pdf"
        assert _is_pdf(report_path)
