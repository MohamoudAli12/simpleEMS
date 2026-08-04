"""Shared pytest configuration and fixtures for the simpleEMS test suite.

Three things happen here that every test depends on:

* matplotlib is forced onto the headless Agg backend *before* anything imports
  ``simpleEMS.sim_tools``, which sets ``plt.rcParams`` at import time. Without
  this, the PyQt6 backend is selected and any ``plot_*`` call blocks on a
  window that never opens in CI.
* rich is pinned to plain text at a fixed width, so what the CLI prints does
  not depend on the terminal the suite happens to run in (see below).
* ``availability`` fixtures skip the tiers that need external binaries
  (openEMS, getdp) or heavyweight optional imports, so the pure-Python tier
  still runs anywhere.
"""

import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

# Every assertion on CLI output is a substring check against text rich has
# rendered, and rich renders differently depending on what it believes about
# the terminal. With colour on it styles the two dashes of an option
# separately, so "--prefix" is never in the output as a literal string; at 80
# columns it folds a long path across a line break, so a tmp-path check fails
# on a narrow terminal and passes on a wide one. A CI runner forces colour on
# and gives 80 columns, which is why those tests pass locally and fail there.
os.environ.pop("FORCE_COLOR", None)
os.environ["NO_COLOR"] = "1"
os.environ["COLUMNS"] = "200"
# typer force-enables colour when GITHUB_ACTIONS (or FORCE_COLOR/PY_COLORS)
# is set, and its force_terminal overrides NO_COLOR, so NO_COLOR alone is not
# enough to make the rendered help plain text.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------
def _importable(name: str) -> bool:
    """Return True if ``name`` imports cleanly."""
    try:
        __import__(name)
    except Exception:
        return False
    return True


HAS_CSXCAD = _importable("CSXCAD") and _importable("openEMS")
HAS_CADQUERY = _importable("cadquery")
HAS_OPENEMS_BIN = shutil.which("openEMS") is not None
HAS_GETDP_BIN = shutil.which("getdp") is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip marked tests whose external dependency is unavailable.

    Tests declare what they need with a marker rather than repeating an
    ``importorskip`` in every body, so ``-m needs_getdp_bin`` also works as a
    selector.
    """
    skips = {
        "needs_csxcad": (HAS_CSXCAD, "CSXCAD/openEMS Python bindings not installed"),
        "needs_cadquery": (HAS_CADQUERY, "cadquery not installed"),
        "needs_openems_bin": (HAS_OPENEMS_BIN, "openEMS binary not on PATH"),
        "needs_getdp_bin": (HAS_GETDP_BIN, "getdp binary not on PATH"),
    }
    for item in items:
        for marker, (available, reason) in skips.items():
            if marker in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))


# ---------------------------------------------------------------------
# Substrate / parameter fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def fr4():
    """Standard FR-4 substrate properties, as a kwargs dict.

    1.6 mm / eps_r 4.4 is the substrate every example in ``examples/`` uses,
    so geometry computed from it is comparable against the published figures.
    """
    return {
        "substrate_eps_r": 4.4,
        "substrate_tand": 0.001,
        "substrate_thickness_mm": 1.6,
        "charac_imp": 50,
    }


@pytest.fixture
def inset_params(fr4):
    """A 2.45 GHz inset-fed patch parameter object on FR-4."""
    from simpleEMS.patch_antenna import InsetFedPatchParams

    return InsetFedPatchParams(resonant_freq=2.45e9, span_freq=0.5e9, **fr4)


@pytest.fixture
def probe_params(fr4):
    """A 2.45 GHz probe-fed patch parameter object on FR-4."""
    from simpleEMS.patch_antenna import ProbeFedPatchParams

    return ProbeFedPatchParams(resonant_freq=2.45e9, span_freq=0.5e9, **fr4)


@pytest.fixture
def mline_params(fr4):
    """A 2.45 GHz quarter-wave microstrip line parameter object on FR-4."""
    from simpleEMS.microstrip_line import MicrostripLineParams

    return MicrostripLineParams(min_freq=1e9, max_freq=4e9, target_freq=2.45e9, **fr4)


@pytest.fixture
def filter_params(fr4):
    """A 3rd-order Butterworth band-stop quarter-wave stub filter on FR-4."""
    from simpleEMS.quarterwave_stub_filter import QuarterWaveFilterParams

    return QuarterWaveFilterParams(
        min_freq=1e9,
        max_freq=4e9,
        centre_freq=2.45e9,
        bandwidth_freq=1.0e9,
        filter_type="bandstop",
        filter_response="butterworth",
        filter_order=3,
        **fr4,
    )


@pytest.fixture
def bandpass_filter_params():
    """A 3rd-order Butterworth band-pass quarter-wave stub filter.

    Taken from ``examples/BandPassQuarterWaveFilter.py``. A band-pass design
    needs low-impedance -- and therefore wide -- stubs, so it does not fit on
    the ``fr4`` substrate the band-stop fixture uses: the stubs would overlap
    and construction raises.
    """
    from simpleEMS.quarterwave_stub_filter import QuarterWaveFilterParams

    return QuarterWaveFilterParams(
        substrate_eps_r=3.3,
        substrate_tand=0.001,
        substrate_thickness_mm=1.6,
        min_freq=0.5e9,
        max_freq=3e9,
        centre_freq=1.5e9,
        bandwidth_freq=1e9,
        filter_type="bandpass",
        filter_response="butterworth",
        filter_order=3,
    )


# ---------------------------------------------------------------------
# Simulation-setup fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def sim_for(request):
    """Factory returning a fresh ``SimSetup`` for a given params object.

    Each call builds a new ``ContinuousStructure``/``openEMS`` pair, so tests
    that mutate geometry never leak into one another.
    """
    from simpleEMS.sim_tools import setup_simulation

    def _make(params, **kwargs):
        return setup_simulation(params, **kwargs)

    return _make


@pytest.fixture
def built_inset(inset_params, sim_for):
    """A fully built (but unsimulated and unmeshed) inset-fed patch antenna.

    Returns ``(antenna, sim, params, port)``.
    """
    from simpleEMS.patch_antenna import InsetFedPatchAntenna

    sim = sim_for(inset_params)
    antenna = InsetFedPatchAntenna(inset_params, sim)
    port = antenna.build_inset_fed_patch_antenna()
    return antenna, sim, inset_params, port


@pytest.fixture
def built_probe(probe_params, sim_for):
    """A fully built (but unsimulated and unmeshed) probe-fed patch antenna.

    Returns ``(antenna, sim, params, port)``.
    """
    from simpleEMS.patch_antenna import ProbeFedPatchAntenna

    sim = sim_for(probe_params)
    antenna = ProbeFedPatchAntenna(probe_params, sim)
    port = antenna.build_probe_fed_patch_antenna()
    return antenna, sim, probe_params, port


@pytest.fixture
def built_mline(mline_params, sim_for):
    """A fully built (but unsimulated and unmeshed) microstrip line.

    Returns ``(line, sim, params, ports)``.
    """
    from simpleEMS.microstrip_line import MicrostripLine

    sim = sim_for(mline_params)
    line = MicrostripLine(mline_params, sim)
    ports = line.build_microstrip_line()
    return line, sim, mline_params, ports


@pytest.fixture
def built_bandstop(filter_params, sim_for):
    """A fully built (but unsimulated and unmeshed) band-stop stub filter.

    Returns ``(structure, sim, params, ports)``.
    """
    from simpleEMS.quarterwave_stub_filter import BandStopQuarterWaveFilter

    sim = sim_for(filter_params)
    structure = BandStopQuarterWaveFilter(filter_params, sim)
    ports = structure.build_band_stop_quarter_wave_filter()
    return structure, sim, filter_params, ports


@pytest.fixture
def built_bandpass(bandpass_filter_params, sim_for):
    """A fully built (but unsimulated and unmeshed) band-pass stub filter.

    Returns ``(structure, sim, params, ports)``.
    """
    from simpleEMS.quarterwave_stub_filter import BandPassQuarterWaveFilter

    sim = sim_for(bandpass_filter_params)
    structure = BandPassQuarterWaveFilter(bandpass_filter_params, sim)
    ports = structure.build_band_pass_quarter_wave_filter()
    return structure, sim, bandpass_filter_params, ports


# ---------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------
@pytest.fixture
def synthetic_resonator():
    """A closed-form single-resonance 1-port S-matrix function of frequency.

    Stands in for a FEM solve in sweep tests: it is analytic, passive, and has
    a sharp feature that a rational interpolant has to actually find.
    """

    def s_at(freq: float, f0: float = 2.45e9, q: float = 40.0) -> np.ndarray:
        # series RLC seen through a matched line -> passive, |S11| <= 1
        detune = q * (freq / f0 - f0 / freq)
        s11 = 1.0 / (1.0 + 1j * detune) - 1.0
        return np.array([[s11]], dtype=complex)

    return s_at


@pytest.fixture(autouse=True)
def _no_stray_writes(tmp_path, monkeypatch):
    """Run every test from a temp cwd.

    Several exporters and ``run_simulation`` default to ``Path.cwd() /
    "Sim_Path"``. Without this, a test that forgets an explicit
    ``output_path`` silently litters the repository.
    """
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figures a test opened.

    ``SimTools`` plotting helpers create figures via the pyplot state machine
    and never close them; without this, a long run accumulates hundreds of
    open figures and matplotlib starts warning.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
