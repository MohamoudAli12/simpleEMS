"""Tests for the plain-data parts of :mod:`simpleEMS.fem_backend`.

``Problem`` and ``FEMOptions`` are ordinary dataclasses, so the properties
that read one off the other can be pinned without a mesh or a solve. The
frequency the mesh is sized at is the one that matters most: it sets both the
element size and the air padding, so getting it from the wrong place is a
silent factor on the cost of every FEM run.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which pytest
# reports as an error rather than a skip.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

# fem_backend imports fem_geometry, which dlopen()s libGLU through gmsh at
# import time; that raises OSError rather than ImportError on a host without
# it, which importorskip does not catch.
try:
    import gmsh  # noqa: F401
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

from simpleEMS.fem_backend import FEMOptions, Problem  # noqa: E402


class TestMeshFrequency:
    def problem(self, **options):
        """A problem swept 1-5 GHz. It carries no solids, because the mesh
        frequency is read off the options and the sweep alone."""
        return Problem(
            step_file="structure.step",
            freqs=np.linspace(1e9, 5e9, 50),
            options=FEMOptions(**options),
        )

    def test_it_falls_back_to_the_top_of_the_sweep(self):
        """``simulate_step_FEM`` and a hand-built ``Problem`` carry no design
        frequency, so they keep meshing at the top of the sweep."""
        assert self.problem().mesh_freq == 5e9

    def test_an_explicit_mesh_freq_is_used(self):
        assert self.problem(mesh_freq=1.5e9).mesh_freq == 1.5e9

    @pytest.mark.parametrize("value", [0.0, -2.45e9], ids=["zero", "negative"])
    def test_a_non_positive_mesh_freq_is_refused(self, value):
        """A wavelength is divided by it, so the mistake has to be caught where
        it is made rather than inside gmsh."""
        with pytest.raises(ValueError, match="mesh_freq must be > 0 Hz"):
            FEMOptions(mesh_freq=value)
