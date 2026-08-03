"""Tests for the scripts in ``examples/``.

The examples are the documentation most users read first, and nothing else in
the suite touches them -- a rename in the public API breaks every one of them
silently until somebody runs one by hand. (That is not hypothetical: commit
"fix: simple fixes of examples" was exactly that.)

Two tiers here:

* Static checks, which run everywhere and take milliseconds. Every example
  must compile, and every name it imports from ``simpleEMS`` or calls on
  ``SimTools`` must actually exist.
* A ``slow`` execution pass that runs each script with the solver and the GUI
  stubbed out, so all the geometry building, plotting and exporting really
  happens. Scripts are copied to a temp directory first, because they write
  their output next to themselves.

This file used to be a ``__main__`` helper with no test functions in it, so
pytest collected nothing from it and the examples were never exercised.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import simpleEMS


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))

# Named so failures say which script broke rather than "example7".
EXAMPLE_IDS = [path.name for path in EXAMPLES]


def parsed(path):
    return ast.parse(path.read_text(), filename=str(path))


def simpleems_imports(tree):
    """Names each ``from simpleEMS import ...`` in ``tree`` pulls in."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "simpleEMS"
        ):
            names.extend(alias.name for alias in node.names)
    return names


def simtools_attributes(tree):
    """Every ``SimTools.<name>`` referenced in ``tree``."""
    return [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "SimTools"
    ]


def test_the_examples_directory_is_found():
    """Guards the glob above: an empty list would make every parametrised test
    below vanish silently rather than fail."""
    assert EXAMPLES, f"no example scripts found under {EXAMPLES_DIR}"


# ---------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------
@pytest.mark.parametrize("path", EXAMPLES, ids=EXAMPLE_IDS)
class TestStatic:
    def test_compiles(self, path):
        compile(path.read_text(), str(path), "exec")

    def test_every_imported_name_is_exported(self, path):
        """The failure mode this catches: a public name is renamed, the
        package still imports, and the example dies on its import line."""
        for name in simpleems_imports(parsed(path)):
            assert name in simpleEMS.__all__, (
                f"{path.name} imports {name!r}, which simpleEMS does not export"
            )

    def test_every_imported_name_resolves(self, path):
        """``__all__`` agreeing is not enough -- the lazy loader has to be able
        to actually produce the object."""
        for name in simpleems_imports(parsed(path)):
            assert getattr(simpleEMS, name) is not None

    def test_simtools_methods_exist(self, path):
        from simpleEMS.sim_tools import SimTools

        for attr in simtools_attributes(parsed(path)):
            assert hasattr(SimTools, attr), (
                f"{path.name} calls SimTools.{attr}, which does not exist"
            )

    def test_imports_only_from_the_public_package(self, path):
        """Examples are documentation; reaching into a submodule shows readers
        a path that is not part of the supported API."""
        for node in ast.walk(parsed(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "simpleEMS."
            ):
                pytest.fail(f"{path.name} imports from {node.module}, not simpleEMS")


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
# Applied inside the subprocess, before the example is executed. What stays
# live is everything the examples exist to demonstrate: parameter derivation,
# geometry construction, meshing, the 2D plots and the exporters. What is
# stubbed is what needs a solver binary, a GPU or a human -- those are covered
# by test_solver_smoke.py and test_radiation_plots.py instead.
_HARNESS = """
import os, runpy, sys

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

from simpleEMS import sim_tools
from simpleEMS.sim_tools import SimData, SimTools


def fake_sim_data(sim=None, port=None, output_path=None, **kwargs):
    freqs = np.linspace(1e9, 3e9, 21)
    s11 = 0.2 * np.exp(1j * np.linspace(0, np.pi, freqs.size))
    s21 = np.sqrt(1 - np.abs(s11) ** 2) * np.exp(
        -1j * np.linspace(0, 4 * np.pi, freqs.size)
    )
    vswr = (1 + np.abs(s11)) / (1 - np.abs(s11))
    z11 = 50 * (1 + s11) / (1 - s11)
    return SimData(
        freqs, s11, s21, z11, vswr, 1.0, np.ones_like(s11), np.ones_like(s11), 50.0
    )


class FakeFarField:
    def __init__(self, theta, phi):
        th = np.deg2rad(np.atleast_1d(np.asarray(theta, dtype=float)))
        ph = np.atleast_1d(np.asarray(phi, dtype=float))
        self.E_norm = np.cos(th / 2.0)[:, None] ** 2 * np.ones((1, ph.size))
        self.Dmax = np.array([4.0])
        self.Prad = np.array([0.75])
        self.P_rad = self.E_norm ** 2
        self.theta = th
        self.phi = np.deg2rad(ph)
        self.Ploss = np.array([0.25])


class FakeNF2FF:
    def CalcNF2FF(self, output_path, freq, theta, phi, **kwargs):
        return FakeFarField(theta, phi)


SimTools.run_simulation = staticmethod(lambda *a, **k: None)
SimTools.write_and_show_structure = staticmethod(lambda *a, **k: None)
SimTools.show_plots = staticmethod(lambda *a, **k: None)
SimTools.compute_sim_data = staticmethod(fake_sim_data)
SimTools.create_nf2ff = staticmethod(lambda *a, **k: FakeNF2FF())
for _name in ("plot_3d_directivity", "plot_3d_gain", "plot_3d_power"):
    setattr(SimTools, _name, staticmethod(lambda *a, **k: None))

# AppCSXCAD, in case an example reaches it another way
sim_tools.subprocess.run = lambda *a, **k: None

runpy.run_path(sys.argv[1], run_name="__main__")
"""


@pytest.fixture
def example_sandbox(tmp_path):
    """Copy an example somewhere writable and hand back a way to run it.

    Two things make the subprocess necessary rather than merely tidy. The
    scripts derive their output directory from ``__file__``, so running them
    in place litters the repository. And running them in-process leaves a
    module namespace full of CSXCAD and openEMS objects to be torn down all at
    once when ``runpy`` discards it; the two libraries then double-free the
    structure they share, which corrupts the heap and makes *unrelated* tests
    later in the session fail with impossible errors (a ``CSPropMaterial``
    reporting its type as ``ProbeBox``). A separate interpreter per example
    contains that, and is how a user runs them anyway.
    """
    harness = tmp_path / "_harness.py"
    harness.write_text(_HARNESS)

    def _run(path, check=True):
        script = tmp_path / path.name
        shutil.copy(path, script)
        for extra in ("structure.xml", "structure.step"):
            source = EXAMPLES_DIR / extra
            if source.exists():
                shutil.copy(source, tmp_path / extra)

        completed = subprocess.run(
            [sys.executable, str(harness), str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if check and completed.returncode != 0:
            pytest.fail(
                f"{path.name} exited {completed.returncode}\n"
                f"--- stdout ---\n{completed.stdout[-3000:]}\n"
                f"--- stderr ---\n{completed.stderr[-3000:]}"
            )
        return completed, tmp_path

    return _run


# These need a real solver run or a prebuilt model to be meaningful, and are
# covered directly elsewhere rather than through the example.
NEEDS_REAL_SOLVER = {
    # loads a prebuilt structure.xml and solves it -- see
    # test_fdtd_standalone_model.py::TestSimulateModelEndToEnd
    "run_model.py",
    # standalone STEP entry points -- see test_fdtd_import_step.py and
    # test_fem_step.py, which drive them directly
    "InsetFedPatch_24GHz_StepFDTD.py",
    "InsetFedPatch_24GHz_StepFEM.py",
}

RUNNABLE = [p for p in EXAMPLES if p.name not in NEEDS_REAL_SOLVER]
RUNNABLE_IDS = [p.name for p in RUNNABLE]


@pytest.mark.slow
@pytest.mark.needs_csxcad
@pytest.mark.parametrize("path", RUNNABLE, ids=RUNNABLE_IDS)
def test_example_runs(path, example_sandbox):
    """Run the script end to end with the solver stubbed.

    A script that fails here is broken for users too -- the stubs only remove
    the solver, the windows and the far-field transform, never the geometry or
    the API calls.
    """
    pytest.importorskip("CSXCAD")
    pytest.importorskip("openEMS")

    example_sandbox(path)


@pytest.mark.slow
@pytest.mark.needs_csxcad
def test_a_runnable_example_actually_builds_geometry(example_sandbox):
    """Guard against the stubs hollowing the run out into a no-op: the script
    must leave real output behind."""
    pytest.importorskip("CSXCAD")

    _completed, sandbox = example_sandbox(EXAMPLES_DIR / "MicrostripLine.py")

    written = list(sandbox.rglob("*"))
    assert any(p.suffix in {".png", ".txt", ".xml", ".stl"} for p in written)


def test_every_example_is_either_run_or_explicitly_excluded():
    """Stops an example being quietly dropped from the run tier by a typo in
    the exclusion set."""
    assert NEEDS_REAL_SOLVER.issubset(EXAMPLE_IDS)
    assert len(RUNNABLE) + len(NEEDS_REAL_SOLVER) == len(EXAMPLES)
