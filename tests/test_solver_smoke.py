"""End-to-end smoke tests that actually invoke the solvers.

These are the only tests that run openEMS or GetDP for real. They deliberately
use a tiny, badly converged model -- a coarse mesh and a few hundred timesteps
-- because the point is to prove the *pipeline* works: geometry is written,
the solver runs, its output files are found and parsed, and the results come
back in the documented shape.

Physical accuracy is explicitly NOT asserted. A model this coarse gives the
wrong answer, and pinning a wrong answer would be worse than pinning nothing.
What is asserted are things that must hold regardless of accuracy: array
shapes, passivity, finiteness, and the derived-quantity identities linking
S11, VSWR, and Z11.

Marked ``slow``; run with ``pytest -m slow`` or deselect with ``-m "not
slow"``.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.needs_csxcad]

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS.microstrip_line import MicrostripLine, MicrostripLineParams  # noqa: E402
from simpleEMS.sim_tools import SimData, setup_simulation, SimTools  # noqa: E402


# Coarse on purpose: 4 cells per wavelength and 300 timesteps runs in well
# under a second while still exercising every stage of the pipeline.
COARSE_FDTD = {
    "num_points": 21,
    "FDTD_timestep": 300,
    "FDTD_mesh_resolution_factor": 4,
    "FDTD_metal_mesh_resolution_factor": 8,
}

COARSE_FEM = {
    "num_points": 21,
    "backend_engine": "FEM",
    "FEM_num_solve_points": 4,
    "FEM_air_pad_mm": 3.0,
    "FEM_elems_per_wavelength": 6.0,
    "FEM_min_layers": 1,
}


def line_params(fr4, **extra):
    return MicrostripLineParams(
        min_freq=2e9, max_freq=3e9, target_freq=2.45e9, **fr4, **extra
    )


def assert_well_formed(data: SimData, expected_points: int, two_port: bool):
    """Invariants any correct backend must satisfy, at any accuracy."""
    assert isinstance(data, SimData)
    assert data.freqs.shape == (expected_points,)
    assert data.s11.shape == (expected_points,)
    assert data.z11.shape == (expected_points,)
    assert data.vswr.shape == (expected_points,)

    assert np.all(np.isfinite(data.s11))
    assert np.all(np.isfinite(data.z11))

    # a passive structure cannot reflect more than it is given
    assert np.all(np.abs(data.s11) <= 1.0 + 1e-6)

    # VSWR is derived from |S11| and is never below 1
    assert np.all(data.vswr >= 1.0)

    if two_port:
        assert data.s21 is not None
        assert data.s21.shape == (expected_points,)
        assert np.all(np.abs(data.s21) <= 1.0 + 1e-6)
    else:
        assert data.s21 is None


# ---------------------------------------------------------------------
# FDTD
# ---------------------------------------------------------------------
@pytest.mark.needs_openems_bin
class TestFdtdPipeline:
    @pytest.fixture(scope="class")
    def solved(self, tmp_path_factory):
        """Run one FDTD solve for the whole class -- it is the expensive bit."""
        fr4 = {
            "substrate_eps_r": 4.4,
            "substrate_tand": 0.001,
            "substrate_thickness_mm": 1.6,
            "charac_imp": 50,
        }
        params = line_params(fr4, **COARSE_FDTD)
        sim = setup_simulation(params)
        line = MicrostripLine(params, sim)
        ports = line.build_microstrip_line()
        line.create_mesh()

        out = tmp_path_factory.mktemp("fdtd_run")
        SimTools.run_simulation(sim, output_path=out)
        data = SimTools.compute_sim_data(sim, ports, output_path=out)
        return sim, params, ports, out, data

    def test_solver_writes_port_probe_files(self, solved):
        """``compute_sim_data`` reads these by name; if the solver wrote them
        elsewhere the parse fails with a confusing IO error."""
        _sim, _params, _ports, out, _data = solved

        names = {path.name for path in out.iterdir()}

        assert "port_ut_1" in names
        assert "port_it_1" in names
        assert "port_ut_2" in names
        assert "port_it_2" in names

    def test_results_are_well_formed(self, solved):
        _sim, params, _ports, _out, data = solved

        assert_well_formed(data, params.num_points, two_port=True)

    def test_frequency_grid_matches_the_setup(self, solved):
        sim, _params, _ports, _out, data = solved

        assert data.freqs == pytest.approx(sim.freqs)

    def test_reference_impedance_is_carried_through(self, solved):
        _sim, params, _ports, _out, data = solved

        assert data.ref_impedance == params.charac_imp

    def test_vswr_agrees_with_s11(self, solved):
        """VSWR = (1+|S11|)/(1-|S11|), with |S11| clipped at 0.999."""
        _sim, _params, _ports, _out, data = solved

        magnitude = np.clip(np.abs(data.s11), 0, 0.999)
        expected = (1 + magnitude) / (1 - magnitude)

        assert data.vswr == pytest.approx(expected, rel=1e-9)

    def test_port_voltage_and_current_are_returned(self, solved):
        _sim, params, _ports, _out, data = solved

        assert data.port_voltage.shape == (params.num_points,)
        assert data.port_current.shape == (params.num_points,)

    def test_impedance_is_voltage_over_current(self, solved):
        _sim, _params, _ports, _out, data = solved

        assert data.z11 == pytest.approx(
            data.port_voltage / data.port_current, rel=1e-9
        )

    def test_input_power_is_real_and_finite(self, solved):
        _sim, _params, _ports, _out, data = solved

        assert np.all(np.isfinite(data.input_power))
        assert np.all(np.isreal(data.input_power))

    def test_a_matched_line_passes_most_of_its_power(self, solved):
        """The loosest possible physics check: a 50 ohm line into a 50 ohm
        port must transmit more than it reflects. This holds even on a mesh
        far too coarse for an accurate answer."""
        _sim, _params, _ports, _out, data = solved

        assert np.mean(np.abs(data.s21)) > np.mean(np.abs(data.s11))

    def test_single_port_call_returns_no_s21(self, solved):
        """Passing one port instead of a list selects the one-port branch."""
        sim, params, ports, out, _data = solved

        data = SimTools.compute_sim_data(sim, ports[0], output_path=out)

        assert_well_formed(data, params.num_points, two_port=False)

    def test_touchstone_export_of_real_results(self, solved, tmp_path):
        import skrf

        _sim, _params, _ports, _out, data = solved

        SimTools.export_touchstone(
            data.freqs, data.s11, s21=data.s21, output_path=tmp_path
        )
        network = skrf.Network(str(tmp_path / "touchstone" / "s_param.s2p"))

        assert network.nports == 2
        assert network.s[:, 0, 0] == pytest.approx(data.s11, abs=1e-9)

    def test_relative_output_path_is_accepted(self, fr4, tmp_path, monkeypatch):
        """``output_path`` is resolved before it reaches ``FDTD.Run``.

        Regression guard: openEMS chdirs into the path and then asserts it
        matches ``os.getcwd()``, so an unresolved relative path always tripped
        its "Current working directory is different from sim_path" check. The
        documented default is absolute, which is why no example hit it.
        """
        monkeypatch.chdir(tmp_path)
        params = line_params(fr4, **COARSE_FDTD)
        sim = setup_simulation(params)
        line = MicrostripLine(params, sim)
        ports = line.build_microstrip_line()
        line.create_mesh()

        Path("relative_out").mkdir()
        SimTools.run_simulation(sim, output_path=Path("relative_out"))

        # results must land in the directory that was asked for
        written = {path.name for path in (tmp_path / "relative_out").iterdir()}
        assert "port_ut_1" in written

        data = SimTools.compute_sim_data(
            sim, ports, output_path=tmp_path / "relative_out"
        )
        assert_well_formed(data, params.num_points, two_port=True)

    def test_default_output_path_works(self, fr4, tmp_path, monkeypatch):
        """The documented default is ``cwd / "Sim_Path"``."""
        monkeypatch.chdir(tmp_path)
        params = line_params(fr4, **COARSE_FDTD)
        sim = setup_simulation(params)
        line = MicrostripLine(params, sim)
        line.build_microstrip_line()
        line.create_mesh()

        SimTools.run_simulation(sim)

        assert (tmp_path / "Sim_Path" / "port_ut_1").exists()


# ---------------------------------------------------------------------
# FEM
# ---------------------------------------------------------------------
@pytest.mark.needs_getdp_bin
class TestFemPipeline:
    @pytest.fixture(scope="class")
    def solved(self, tmp_path_factory):
        fr4 = {
            "substrate_eps_r": 4.4,
            "substrate_tand": 0.001,
            "substrate_thickness_mm": 1.6,
            "charac_imp": 50,
        }
        params = line_params(fr4, **COARSE_FEM)
        sim = setup_simulation(params)
        line = MicrostripLine(params, sim)
        ports = line.build_microstrip_line()

        out = tmp_path_factory.mktemp("fem_run")
        SimTools.run_simulation(sim, output_path=out)
        data = SimTools.compute_sim_data(sim, ports, output_path=out)
        return sim, params, ports, out, data

    def test_pipeline_writes_its_intermediate_files(self, solved):
        """The FEM stages hand state between each other through files rather
        than Python objects; each must appear."""
        _sim, _params, _ports, out, _data = solved

        names = {path.name for path in out.iterdir()}

        assert "structure.step" in names
        assert "structure.msh" in names
        assert "structure.pro" in names
        assert "fem_mesh.json" in names
        assert "fem_sparams.npz" in names

    def test_generated_pro_is_non_trivial(self, solved):
        _sim, _params, _ports, out, _data = solved

        content = (out / "structure.pro").read_text()

        assert "Resolution" in content
        assert "Get_SParameters" in content

    def test_results_are_well_formed(self, solved):
        _sim, params, _ports, _data_out, data = solved

        assert_well_formed(data, params.num_points, two_port=True)

    def test_port_argument_is_ignored_by_the_fem_branch(self, solved):
        """Documented: the FEM path reads its results from disk, so the port
        object passed in must not change the answer."""
        sim, _params, ports, out, data = solved

        again = SimTools.compute_sim_data(sim, ports[0], output_path=out)

        assert again.s11 == pytest.approx(data.s11)

    def test_frequency_grid_matches_the_setup(self, solved):
        sim, _params, _ports, _out, data = solved

        assert data.freqs == pytest.approx(sim.freqs)

    def test_solve_budget_is_respected(self, solved):
        """The whole economy of the FEM backend: 4 solves, 21 output points."""
        _sim, params, _ports, _out, data = solved

        assert params.FEM_num_solve_points == 4
        assert len(data.freqs) == 21

    def test_mesh_is_reused_when_nothing_changed(self, solved):
        """A second run over an unchanged geometry must not re-mesh; the
        fingerprint in ``fem_mesh.json`` is what prevents it."""
        sim, _params, _ports, out, _data = solved

        before = (out / "structure.msh").stat().st_mtime_ns
        SimTools.run_simulation(sim, output_path=out)
        after = (out / "structure.msh").stat().st_mtime_ns

        assert before == after

    def test_a_matched_line_passes_most_of_its_power(self, solved):
        _sim, _params, _ports, _out, data = solved

        assert np.mean(np.abs(data.s21)) > np.mean(np.abs(data.s11))


# ---------------------------------------------------------------------
# Cross-backend
# ---------------------------------------------------------------------
@pytest.mark.needs_openems_bin
@pytest.mark.needs_getdp_bin
class TestBackendAgreement:
    def test_both_backends_return_the_same_shape(self, fr4, tmp_path):
        """The FEM backend's contract is that it is drop-in compatible with
        the FDTD one -- same ``SimData``, so the same plotting and export code
        works unchanged. Values are not compared; the meshes are far too
        coarse for that."""
        results = {}
        for label, extra in (("fdtd", COARSE_FDTD), ("fem", COARSE_FEM)):
            params = line_params(fr4, **extra)
            sim = setup_simulation(params)
            line = MicrostripLine(params, sim)
            ports = line.build_microstrip_line()
            if params.backend_engine == "FDTD":
                line.create_mesh()

            out = tmp_path / label
            out.mkdir()
            SimTools.run_simulation(sim, output_path=out)
            results[label] = SimTools.compute_sim_data(sim, ports, output_path=out)

        assert results["fdtd"].s11.shape == results["fem"].s11.shape
        assert results["fdtd"].freqs == pytest.approx(results["fem"].freqs)
        assert type(results["fdtd"]) is type(results["fem"])
