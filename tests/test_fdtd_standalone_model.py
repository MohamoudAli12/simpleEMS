"""Tests for the standalone-model workflow.

``fdtd_standalone_model`` is the path a user takes when the geometry did not
come from a simpleEMS builder: load someone else's ``structure.xml``, bolt an
FDTD setup onto it, and post-process the result. Almost all of it is XML and
CSXCAD-property bookkeeping, so most of this file needs no solver -- only the
end-to-end tests at the bottom are marked ``slow``.

The contract being pinned is the round trip: what ``add_fdtd_setup`` writes,
``get_freq_range`` must be able to read back, and ``reconstruct_ports`` must
recover the same ports openEMS originally wrote into the file.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from CSXCAD import ContinuousStructure  # noqa: E402

from simpleEMS.fdtd_standalone_model import (  # noqa: E402
    add_fdtd_setup,
    get_freq_range,
    reconstruct_ports,
    simulate_model,
)
from simpleEMS.microstrip_line import MicrostripLine, MicrostripLineParams  # noqa: E402
from simpleEMS.sim_tools import SimData, SimSetup, setup_simulation  # noqa: E402


# Coarse on purpose -- see the note in test_solver_smoke.py. 300 timesteps on
# a 4-cell-per-wavelength grid runs in under a second.
COARSE = {
    "num_points": 21,
    "FDTD_timestep": 300,
    "FDTD_mesh_resolution_factor": 4,
    "FDTD_metal_mesh_resolution_factor": 8,
}


# The round-trip assertions below read the written XML rather than loading it
# back through ``openEMS.GetCSX()``, for a measured reason. An earlier version
# of this file had a fixture that returned ``fdtd.GetCSX()`` and let the
# ``fdtd`` object go out of scope; reading a property off the result
# segfaulted the interpreter outright. Keeping the handle only while its owner
# is alive avoids the crash, but this module still failed roughly four runs in
# eight, always with a CSXCAD property whose Python wrapper class disagreed
# with its own ``GetTypeString()`` -- a ``CSPropMetal`` reporting itself as an
# ``Excitation``, then dying on a missing method. Rewriting these assertions
# to read the file instead took that to zero.
#
# Note that the library's own ``GetCSX()`` calls, which use the handle
# immediately while the owner is alive, were measured and are *not* implicated:
# swapping them out changed the residual failure rate not at all (3/15 either
# way, under coverage). Asserting against the file is also the better test on
# its own merits -- it checks the persisted bytes, which is what the next run
# actually loads.


def written_property_names(path):
    """Names of the properties recorded in a written CSX or openEMS XML."""
    root = ET.parse(path).getroot()
    structure = (
        root if root.tag == "ContinuousStructure" else root.find("ContinuousStructure")
    )
    return [prop.get("Name") for prop in structure.find("Properties")]


def written_grid_lines(path):
    """Mesh lines recorded in a written CSX or openEMS XML, per axis."""
    root = ET.parse(path).getroot()
    structure = (
        root if root.tag == "ContinuousStructure" else root.find("ContinuousStructure")
    )
    grid = structure.find("RectilinearGrid")
    return [np.fromstring(grid.find(f"{axis}Lines").text, sep=",") for axis in "XYZ"]


@pytest.fixture
def meshed_line(fr4):
    """A built and meshed two-port microstrip line.

    Returns ``(sim, params)``. This is the stand-in for "a model that came
    from somewhere else": everything below writes it out and reloads it
    through the public entry points rather than reaching into ``sim``.
    """
    params = MicrostripLineParams(
        min_freq=2e9, max_freq=3e9, target_freq=2.45e9, **fr4, **COARSE
    )
    sim = setup_simulation(params)
    line = MicrostripLine(params, sim)
    line.build_microstrip_line()
    line.create_mesh()
    return sim, params


@pytest.fixture
def csx_xml(meshed_line, tmp_path):
    """A geometry-only model: root ``<ContinuousStructure>``, no FDTD setup."""
    sim, _params = meshed_line
    path = tmp_path / "structure.xml"
    sim.CSX.Write2XML(str(path))
    return path


@pytest.fixture
def openems_xml(meshed_line, tmp_path):
    """A complete model: root ``<openEMS>``, FDTD setup included."""
    sim, _params = meshed_line
    path = tmp_path / "model.xml"
    sim.FDTD.Write2XML(str(path))
    return path


@pytest.fixture
def no_gui(monkeypatch):
    """Stop ``write_and_show_structure`` from launching AppCSXCAD.

    ``simulate_model`` calls it unconditionally to dump the loaded geometry,
    and the real call blocks on a GUI window that never opens in CI.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr("simpleEMS.sim_tools.subprocess.run", fake_run)
    return calls


# ---------------------------------------------------------------------
# get_freq_range
# ---------------------------------------------------------------------
class TestGetFreqRange:
    def test_reads_back_the_excitation_band(self, openems_xml, meshed_line):
        """``setup_simulation`` centres the Gaussian on the requested band, so
        the values recovered here are that band's centre and half-width."""
        _sim, params = meshed_line

        f0, fc = get_freq_range(openems_xml)

        assert f0 == pytest.approx(0.5 * (params.min_freq + params.max_freq))
        assert fc == pytest.approx(0.5 * (params.max_freq - params.min_freq))

    def test_accepts_a_string_path(self, openems_xml):
        assert get_freq_range(str(openems_xml)) == get_freq_range(openems_xml)

    def test_geometry_only_model_is_rejected(self, csx_xml):
        """The common mistake: passing a CSX.Write2XML() file straight in."""
        with pytest.raises(ValueError, match="no openEMS FDTD setup"):
            get_freq_range(csx_xml)

    def test_the_error_points_at_add_fdtd_setup(self, csx_xml):
        with pytest.raises(ValueError, match="add_fdtd_setup"):
            get_freq_range(csx_xml)

    def test_missing_excitation_element_is_rejected(self, openems_xml, tmp_path):
        tree = ET.parse(openems_xml)
        fdtd = tree.getroot().find("FDTD")
        fdtd.remove(fdtd.find("Excitation"))
        stripped = tmp_path / "no_excitation.xml"
        tree.write(stripped)

        with pytest.raises(ValueError, match="no <Excitation> element"):
            get_freq_range(stripped)

    def test_non_gaussian_excitation_is_rejected(self, csx_xml, tmp_path):
        """A sinusoidal excitation carries no bandwidth, so no post-processing
        band can be derived from it."""
        model = add_fdtd_setup(
            csx_xml,
            freq_range=(2e9, 3e9),
            output_xml_path=tmp_path / "sinus.xml",
            excitation="sinus",
        )

        with pytest.raises(ValueError, match="not a Gaussian pulse"):
            get_freq_range(model)

    def test_the_non_gaussian_error_offers_a_way_out(self, csx_xml, tmp_path):
        model = add_fdtd_setup(
            csx_xml,
            freq_range=(2e9, 3e9),
            output_xml_path=tmp_path / "step.xml",
            excitation="step",
        )

        with pytest.raises(ValueError, match="explicit freqs="):
            get_freq_range(model)


# ---------------------------------------------------------------------
# add_fdtd_setup
# ---------------------------------------------------------------------
class TestAddFdtdSetup:
    def test_writes_an_openems_rooted_document(self, csx_xml, tmp_path):
        out = add_fdtd_setup(
            csx_xml, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
        )

        assert out.exists()
        assert ET.parse(out).getroot().tag == "openEMS"

    def test_returns_the_path_it_wrote(self, csx_xml, tmp_path):
        dst = tmp_path / "explicit.xml"

        assert (
            add_fdtd_setup(csx_xml, freq_range=(2e9, 3e9), output_xml_path=dst) == dst
        )

    def test_the_band_round_trips_through_get_freq_range(self, csx_xml, tmp_path):
        add_fdtd_setup(
            csx_xml, freq_range=(1e9, 5e9), output_xml_path=tmp_path / "band.xml"
        )

        f0, fc = get_freq_range(tmp_path / "band.xml")

        assert f0 == pytest.approx(3e9)
        assert fc == pytest.approx(2e9)

    def test_default_output_path_is_sim_path_under_cwd(self, csx_xml, tmp_path):
        """``tmp_path`` is the cwd -- conftest's ``_no_stray_writes`` chdirs."""
        out = add_fdtd_setup(csx_xml, freq_range=(2e9, 3e9))

        assert out == Path.cwd() / "Sim_Path" / "structure_fdtd.xml"
        assert out.exists()

    def test_creates_the_output_directory(self, csx_xml, tmp_path):
        dst = tmp_path / "deep" / "nested" / "out.xml"

        add_fdtd_setup(csx_xml, freq_range=(2e9, 3e9), output_xml_path=dst)

        assert dst.exists()

    def test_geometry_survives_the_round_trip(self, csx_xml, tmp_path):
        out = add_fdtd_setup(
            csx_xml, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
        )

        assert written_property_names(out) == written_property_names(csx_xml)

    def test_mesh_survives_the_round_trip(self, csx_xml, meshed_line, tmp_path):
        sim, _params = meshed_line
        before = [np.asarray(sim.CSX.GetGrid().GetLines(d)) for d in range(3)]

        out = add_fdtd_setup(
            csx_xml, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
        )

        for dim, expected in enumerate(before):
            assert written_grid_lines(out)[dim] == pytest.approx(expected)

    def test_boundary_conditions_reach_the_xml(self, csx_xml, tmp_path):
        out = add_fdtd_setup(
            csx_xml,
            freq_range=(2e9, 3e9),
            output_xml_path=tmp_path / "bc.xml",
            FDTD_boundary=["MUR"] * 6,
        )

        bc = ET.parse(out).getroot().find("./FDTD/BoundaryCond")
        assert set(bc.attrib.values()) == {"MUR"}

    def test_default_boundary_is_eight_cell_pml(self, csx_xml, tmp_path):
        out = add_fdtd_setup(
            csx_xml, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "bc.xml"
        )

        bc = ET.parse(out).getroot().find("./FDTD/BoundaryCond")
        assert set(bc.attrib.values()) == {"PML_8"}

    def test_run_limits_reach_the_xml(self, csx_xml, tmp_path):
        out = add_fdtd_setup(
            csx_xml,
            freq_range=(2e9, 3e9),
            output_xml_path=tmp_path / "limits.xml",
            FDTD_timestep=1234,
            FDTD_end_criteria=1e-3,
        )

        fdtd = ET.parse(out).getroot().find("FDTD")
        assert int(fdtd.get("NumberOfTimesteps")) == 1234
        assert float(fdtd.get("endCriteria")) == pytest.approx(1e-3)

    @pytest.mark.parametrize("excitation", ["gauss", "sinus", "dirac", "step"])
    def test_every_documented_excitation_is_accepted(
        self, csx_xml, tmp_path, excitation
    ):
        out = add_fdtd_setup(
            csx_xml,
            freq_range=(2e9, 3e9),
            output_xml_path=tmp_path / f"{excitation}.xml",
            excitation=excitation,
        )

        assert ET.parse(out).getroot().find("./FDTD/Excitation") is not None

    def test_unknown_excitation_is_rejected(self, csx_xml, tmp_path):
        with pytest.raises(ValueError, match="excitation must be"):
            add_fdtd_setup(
                csx_xml,
                freq_range=(2e9, 3e9),
                output_xml_path=tmp_path / "bad.xml",
                excitation="triangle",
            )

    def test_existing_output_is_not_clobbered(self, csx_xml, tmp_path):
        dst = tmp_path / "taken.xml"
        dst.write_text("precious")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            add_fdtd_setup(csx_xml, freq_range=(2e9, 3e9), output_xml_path=dst)

        assert dst.read_text() == "precious"

    def test_overwrite_replaces_an_existing_file(self, csx_xml, tmp_path):
        dst = tmp_path / "taken.xml"
        dst.write_text("precious")

        add_fdtd_setup(
            csx_xml, freq_range=(2e9, 3e9), output_xml_path=dst, overwrite=True
        )

        assert ET.parse(dst).getroot().tag == "openEMS"

    @pytest.mark.parametrize(
        "freq_range", [(3e9, 2e9), (2e9, 2e9), (-1e9, 2e9), (0, 0)]
    )
    def test_non_increasing_band_is_rejected(self, csx_xml, tmp_path, freq_range):
        with pytest.raises(ValueError, match="0 <= fmin < fmax"):
            add_fdtd_setup(
                csx_xml, freq_range=freq_range, output_xml_path=tmp_path / "bad.xml"
            )

    def test_unexpected_root_element_is_rejected(self, tmp_path):
        junk = tmp_path / "junk.xml"
        junk.write_text("<NotAStructure/>")

        with pytest.raises(ValueError, match="unexpected root element"):
            add_fdtd_setup(
                junk, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
            )

    def test_unmeshed_model_is_rejected(self, fr4, tmp_path):
        """``add_fdtd_setup`` supplies solver settings only; it will not mesh."""
        params = MicrostripLineParams(
            min_freq=2e9, max_freq=3e9, target_freq=2.45e9, **fr4, **COARSE
        )
        sim = setup_simulation(params)
        MicrostripLine(params, sim).build_microstrip_line()  # no create_mesh
        src = tmp_path / "unmeshed.xml"
        sim.CSX.Write2XML(str(src))

        with pytest.raises(ValueError, match="no mesh lines"):
            add_fdtd_setup(
                src, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
            )

    def test_portless_model_is_rejected(self, tmp_path):
        """Without ports there is nothing to compute network parameters from."""
        csx = ContinuousStructure()
        metal = csx.AddMetal("plate")
        metal.AddBox([0, 0, 0], [10, 10, 1])
        grid = csx.GetGrid()
        grid.SetDeltaUnit(1e-3)
        for axis in "xyz":
            grid.AddLine(axis, [-5, 0, 5, 10, 15])
        src = tmp_path / "portless.xml"
        csx.Write2XML(str(src))

        with pytest.raises(RuntimeError, match="no port_resist_<N> properties"):
            add_fdtd_setup(
                src, freq_range=(2e9, 3e9), output_xml_path=tmp_path / "out.xml"
            )

    def test_an_openems_file_can_be_rebanded(self, openems_xml, tmp_path):
        """Feeding an already-complete model back in replaces its FDTD section,
        which is how you re-band a model without rebuilding its geometry."""
        out = add_fdtd_setup(
            openems_xml,
            freq_range=(10e9, 20e9),
            output_xml_path=tmp_path / "rebanded.xml",
        )

        f0, fc = get_freq_range(out)
        assert f0 == pytest.approx(15e9)
        assert fc == pytest.approx(5e9)

    def test_rebanding_keeps_the_geometry(self, openems_xml, tmp_path):
        out = add_fdtd_setup(
            openems_xml,
            freq_range=(10e9, 20e9),
            output_xml_path=tmp_path / "rebanded.xml",
        )

        assert written_property_names(out) == written_property_names(openems_xml)

    def test_rebanding_keeps_the_mesh(self, openems_xml, tmp_path):
        out = add_fdtd_setup(
            openems_xml,
            freq_range=(10e9, 20e9),
            output_xml_path=tmp_path / "rebanded.xml",
        )

        for dim in range(3):
            assert written_grid_lines(out)[dim] == pytest.approx(
                written_grid_lines(openems_xml)[dim]
            )


# ---------------------------------------------------------------------
# reconstruct_ports
# ---------------------------------------------------------------------
class TestReconstructPorts:
    @pytest.fixture
    def reloaded_csx(self, csx_xml):
        """A structure read from disk, as ``reconstruct_ports`` gets it.

        Built straight from the geometry-only XML rather than by pulling the
        CSX back out of an ``openEMS`` object. Both routes produce the same
        properties, and ``simulate_model`` hands ``reconstruct_ports`` a
        freshly-read structure of exactly this kind -- but the ``openEMS``
        route additionally leaves a second C++ owner of the same structure
        alive, and reading properties off the result then intermittently
        returns garbage (a ``CSPropExcitation`` reporting its type as
        ``ProbeBox``, and so on). Keeping the fixture to a single owner makes
        the test deterministic without making it less faithful.
        """
        csx = ContinuousStructure()
        csx.ReadFromXML(str(csx_xml))
        return csx

    def test_recovers_both_ports(self, reloaded_csx):
        ports, _r = reconstruct_ports(reloaded_csx)

        assert len(ports) == 2

    def test_ports_come_back_sorted_by_number(self, reloaded_csx):
        ports, _r = reconstruct_ports(reloaded_csx)

        assert [p.number for p in ports] == [1, 2]

    def test_reference_impedance_is_the_port_resistance(self, reloaded_csx, fr4):
        _ports, r = reconstruct_ports(reloaded_csx)

        assert r == pytest.approx(fr4["charac_imp"])

    def test_every_port_carries_the_resistance(self, reloaded_csx, fr4):
        ports, _r = reconstruct_ports(reloaded_csx)

        expected = [fr4["charac_imp"]] * len(ports)
        assert [port.R for port in ports] == pytest.approx(expected)
        assert [port.Z_ref for port in ports] == pytest.approx(expected)

    def test_only_the_first_port_is_excited(self, reloaded_csx):
        """``build_microstrip_line`` drives port 1 and terminates port 2."""
        ports, _r = reconstruct_ports(reloaded_csx)

        assert ports[0].excite == 1
        assert ports[1].excite == 0

    def test_probe_filenames_are_recovered(self, reloaded_csx):
        """``CalcPort`` reads the solver output by these names."""
        ports, _r = reconstruct_ports(reloaded_csx)

        for index, port in enumerate(ports, start=1):
            assert port.U_filenames == [f"port_ut_{index}"]
            assert port.I_filenames == [f"port_it_{index}"]

    def test_excitation_axis_is_the_lumped_element_direction(self, reloaded_csx):
        """The port bridges ground to trace, so it is excited along z."""
        ports, _r = reconstruct_ports(reloaded_csx)

        for port in ports:
            assert port.exc_ny == 2

    def test_direction_sign_is_normalised(self, reloaded_csx):
        ports, _r = reconstruct_ports(reloaded_csx)

        for port in ports:
            assert port.direction in (-1.0, 1.0)

    def test_port_extent_matches_the_lumped_element(self, reloaded_csx):
        ports, _r = reconstruct_ports(reloaded_csx)

        for port in ports:
            assert np.all(np.asarray(port.stop) >= np.asarray(port.start))

    def test_empty_structure_yields_no_ports(self):
        ports, r = reconstruct_ports(ContinuousStructure())

        assert ports == []
        assert r == 0.0

    def test_properties_without_a_trailing_number_are_ignored(self):
        csx = ContinuousStructure()
        csx.AddMetal("ground")
        csx.AddMaterial("substrate")

        ports, r = reconstruct_ports(csx)

        assert ports == []
        assert r == 0.0

    def test_a_lumped_element_without_primitives_is_skipped(self):
        """A property can exist with no geometry attached; it has no extent to
        build a port from."""
        csx = ContinuousStructure()
        csx.AddLumpedElement("port_resist_1", ny="z", caps=True, R=50)

        ports, r = reconstruct_ports(csx)

        assert ports == []
        # R is still reported from the last element inspected
        assert r == pytest.approx(50)


# ---------------------------------------------------------------------
# simulate_model
# ---------------------------------------------------------------------
class TestSimulateModel:
    def test_portless_model_is_rejected(self, tmp_path, no_gui):
        """Reached only after the geometry loads, so it needs a real file."""
        csx = ContinuousStructure()
        csx.AddMetal("plate").AddBox([0, 0, 0], [10, 10, 1])
        grid = csx.GetGrid()
        grid.SetDeltaUnit(1e-3)
        for axis in "xyz":
            grid.AddLine(axis, [-5, 0, 5, 10, 15])

        from openEMS.openEMS import openEMS

        fdtd = openEMS(NrTS=10, EndCriteria=1e-3)
        fdtd.SetCSX(csx)
        fdtd.SetBoundaryCond(["PML_8"] * 6)
        fdtd.SetGaussExcite(2.5e9, 0.5e9)
        src = tmp_path / "portless.xml"
        fdtd.Write2XML(str(src))

        with pytest.raises(RuntimeError, match="No ports found"):
            simulate_model(src, output_path=tmp_path / "out", run=False)

    def test_missing_band_and_freqs_is_rejected(self, csx_xml, tmp_path, no_gui):
        """A geometry-only file has no excitation to derive the band from."""
        with pytest.raises(ValueError, match="no openEMS FDTD setup"):
            simulate_model(csx_xml, output_path=tmp_path / "out", run=False)


@pytest.mark.slow
@pytest.mark.needs_openems_bin
class TestSimulateModelEndToEnd:
    """The real thing: write a model out, load it back, solve it.

    Accuracy is not asserted -- see the note at the top of
    ``test_solver_smoke.py``. What matters is that a model that made a full
    round trip through XML still solves and post-processes.
    """

    @pytest.fixture(scope="class")
    def solved(self, tmp_path_factory, request):
        fr4 = {
            "substrate_eps_r": 4.4,
            "substrate_tand": 0.001,
            "substrate_thickness_mm": 1.6,
            "charac_imp": 50,
        }
        params = MicrostripLineParams(
            min_freq=2e9, max_freq=3e9, target_freq=2.45e9, **fr4, **COARSE
        )
        sim = setup_simulation(params)
        line = MicrostripLine(params, sim)
        line.build_microstrip_line()
        line.create_mesh()

        out = tmp_path_factory.mktemp("standalone")
        model = out / "model.xml"
        sim.FDTD.Write2XML(str(model))

        # The GUI launcher has to be stubbed for the whole class; monkeypatch
        # is function-scoped, so patch by hand and undo at teardown.
        import simpleEMS.sim_tools as sim_tools

        original = sim_tools.subprocess.run
        sim_tools.subprocess.run = lambda cmd, **kw: None
        request.addfinalizer(lambda: setattr(sim_tools.subprocess, "run", original))

        data, loaded, charac_imp = simulate_model(model, output_path=out)
        return data, loaded, charac_imp, out, params

    def test_returns_the_documented_triple(self, solved):
        data, loaded, charac_imp, _out, _params = solved

        assert isinstance(data, SimData)
        assert isinstance(loaded, SimSetup)
        assert isinstance(charac_imp, float)

    def test_reference_impedance_comes_from_the_model(self, solved):
        _data, _loaded, charac_imp, _out, _params = solved

        assert charac_imp == pytest.approx(50)

    def test_post_processing_grid_defaults_to_a_thousand_points(self, solved):
        """``num_points`` here is ``simulate_model``'s own default, not the
        builder's: a loaded model carries no ``SimParams`` to inherit it
        from, only the excitation band its XML records."""
        data, _loaded, _charac_imp, _out, _params = solved

        assert data.freqs.shape == (1000,)

    def test_results_are_well_formed(self, solved):
        data, _loaded, _charac_imp, _out, _params = solved

        assert data.s11.shape == data.freqs.shape
        assert np.all(np.isfinite(data.s11))
        assert np.all(np.abs(data.s11) <= 1.0 + 1e-6)
        assert np.all(data.vswr >= 1.0)

    def test_two_port_model_yields_s21(self, solved):
        data, _loaded, _charac_imp, _out, _params = solved

        assert data.s21 is not None
        assert data.s21.shape == data.freqs.shape
        assert np.all(np.abs(data.s21) <= 1.0 + 1e-6)

    def test_band_is_derived_from_the_models_own_excitation(self, solved):
        data, _loaded, _charac_imp, _out, params = solved

        assert data.freqs[0] == pytest.approx(params.min_freq)
        assert data.freqs[-1] == pytest.approx(params.max_freq)

    def test_the_solver_wrote_its_probe_files(self, solved):
        _data, _loaded, _charac_imp, out, _params = solved

        names = {path.name for path in out.iterdir()}
        assert {"port_ut_1", "port_it_1", "port_ut_2", "port_it_2"} <= names

    def test_rerun_with_run_false_reuses_the_existing_results(self, solved, no_gui):
        """``run=False`` is the post-process-only path: same numbers, no solve."""
        data, _loaded, _charac_imp, out, _params = solved

        again, _sim, _z = simulate_model(out / "model.xml", output_path=out, run=False)

        assert again.s11 == pytest.approx(data.s11)

    def test_explicit_freqs_override_the_models_band(self, solved, no_gui):
        _data, _loaded, _charac_imp, out, _params = solved
        band = np.linspace(2.2e9, 2.6e9, 7)

        data, _sim, _z = simulate_model(
            out / "model.xml", output_path=out, run=False, freqs=band
        )

        assert data.freqs == pytest.approx(band)
        assert data.s11.shape == (7,)

    def test_num_points_controls_the_grid_size(self, solved, no_gui):
        _data, _loaded, _charac_imp, out, _params = solved

        data, _sim, _z = simulate_model(
            out / "model.xml", output_path=out, run=False, num_points=13
        )

        assert data.freqs.shape == (13,)
