"""Tests for :mod:`simpleEMS.fem_formulation` -- the GetDP ``.pro`` writer.

This module is 489 lines of DSL templating and is excluded from ruff, so
nothing else in the toolchain looks at it. A malformed ``.pro`` fails inside
GetDP with an error that points at the generated file rather than at the
Python that produced it, which makes a golden file the cheapest way to notice
a regression.

``Problem`` and ``Mesh`` are plain dataclasses, so the fixtures below build
them directly. No gmsh run, no solve -- the writer only reads region tags and
material values off them.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.needs_csxcad

# Without CSXCAD/openEMS these imports fail at collection time, which
# pytest reports as an error rather than a skip. importorskip makes the
# whole module skip cleanly; the marker above keeps it selectable with -m.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

from simpleEMS import fem_formulation  # noqa: E402
from simpleEMS.fem_backend import (  # noqa: E402
    FEMOptions,
    PortSpec,
    Problem,
    SolidSpec,
)
from simpleEMS.fem_geometry import Mesh, PortMesh  # noqa: E402
from simpleEMS.fem_materials import (  # noqa: E402
    ABC,
    AIR,
    PEC,
    Dielectric,
    dielectric_region,
    port_region,
)


GOLDEN = Path(__file__).parent / "golden" / "one_port_silver_muller.pro"


def make_problem(**overrides):
    """A one-port patch-like problem: one dielectric, one PEC, one port."""
    defaults = {
        "step_file": "structure.step",
        "name": "structure",
        "solids": {
            "substrate": SolidSpec("substrate", "dielectric", Dielectric(4.4, 0.001)),
            "patch": SolidSpec("patch", "pec"),
            "port_1": SolidSpec("port_1", "port"),
        },
        "ports": [PortSpec("port_1", 1, 50.0, "z")],
        "freqs": np.array([2.45e9]),
        "options": FEMOptions(),
    }
    defaults.update(overrides)
    return Problem(**defaults)


def make_mesh(**overrides):
    """The mesh metadata matching :func:`make_problem`."""
    defaults = {
        "msh_path": "structure.msh",
        "dielectric_regions": {"substrate": dielectric_region(0)},
        "air_region": AIR,
        "pec_region": PEC,
        "port_regions": {
            1: PortMesh(
                number=1,
                region=port_region(1),
                direction="z",
                z0=50.0,
                gap=1.6e-3,
                width=3.0e-3,
                center=(0.0, 0.0, 0.0008),
            )
        },
        "abc_region": ABC,
        "boundary": "silver_muller",
        "bbox": (-0.05, -0.05, 0.0, 0.05, 0.05, 0.0016),
        "box_bbox": (-0.08, -0.08, -0.03, 0.08, 0.08, 0.03),
        "lambda_min": 0.05,
    }
    defaults.update(overrides)
    return Mesh(**defaults)


def write(problem, mesh, workdir) -> str:
    return Path(fem_formulation.write_problem(problem, mesh, workdir)).read_text()


def resolution_block(content: str, name: str) -> str:
    """Return the body of the named ``Resolution`` entry.

    The file defines two resolutions -- ``Analysis`` (all ports at one
    frequency) and ``AnalysisSinglePort`` -- so counting operations across the
    whole file conflates them.
    """
    marker = "\n  { Name "  # two-space indent = a top-level entry in the block
    start = content.index(f"{marker}{name} ;")
    rest = content[start + 1 :]
    end = rest.find(marker)
    return rest if end == -1 else rest[:end]


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------
class TestHelpers:
    def test_fmt_emits_a_decimal_point(self):
        """GetDP parses a bare ``4`` as an integer; the trailing ``.0`` is
        what keeps the material value floating point."""
        assert fem_formulation._fmt(4) == "4.0"
        assert fem_formulation._fmt(4.4) == "4.4"

    def test_fmt_round_trips_exactly(self):
        for value in (1.0, 4.4, 1e-12, 2.45e9, -0.5):
            assert float(fem_formulation._fmt(value)) == value

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("x", "Vector[1., 0., 0.]"),
            ("y", "Vector[0., 1., 0.]"),
            ("z", "Vector[0., 0., 1.]"),
        ],
    )
    def test_dir_vector(self, direction, expected):
        assert fem_formulation._dir_vector(direction) == expected

    def test_unknown_direction_raises(self):
        with pytest.raises(KeyError):
            fem_formulation._dir_vector("w")


# ---------------------------------------------------------------------
# Golden file
# ---------------------------------------------------------------------
class TestGoldenFile:
    def test_matches_the_golden_pro(self, tmp_path):
        """Regenerate the golden only alongside a deliberate formulation
        change, and re-run a real FEM solve when you do."""
        assert write(make_problem(), make_mesh(), tmp_path) == GOLDEN.read_text()

    def test_output_is_reproducible(self, tmp_path):
        first = write(make_problem(), make_mesh(), tmp_path)
        second = write(make_problem(), make_mesh(), tmp_path)

        assert first == second

    def test_contains_no_absolute_paths(self, tmp_path):
        """Everything is written relative to ``myDir``, so the .pro stays
        valid if the working directory moves."""
        content = write(make_problem(), make_mesh(), tmp_path)

        assert str(tmp_path) not in content


# ---------------------------------------------------------------------
# write_problem
# ---------------------------------------------------------------------
class TestWriteProblem:
    def test_writes_a_pro_named_after_the_problem(self, tmp_path):
        path = fem_formulation.write_problem(make_problem(), make_mesh(), tmp_path)

        assert Path(path).name == "structure.pro"
        assert Path(path).is_file()

    def test_problem_name_drives_the_filename(self, tmp_path):
        path = fem_formulation.write_problem(
            make_problem(name="patch_2450"), make_mesh(), tmp_path
        )

        assert Path(path).name == "patch_2450.pro"

    def test_region_tags_match_the_mesh(self, tmp_path):
        """The integers in the .pro are the only link to the mesh; if they
        drift, GetDP solves an empty region and returns zeros."""
        content = write(make_problem(), make_mesh(), tmp_path)

        assert f"Diel_substrate = Region[{dielectric_region(0)}];" in content
        assert f"Air  = Region[{AIR}];" in content
        assert f"Pec  = Region[{PEC}];" in content
        assert f"Port_1 = Region[{port_region(1)}];" in content
        assert f"Abc  = Region[{ABC}];" in content

    def test_permittivity_uses_the_negative_loss_convention(self, tmp_path):
        """The .pro is written for exp(+iwt) (GetDP's native convention), so a
        passive lossy dielectric needs Im[eps] < 0 -- the same sign as
        ``Dielectric.eps_complex``. Getting this backwards makes the material
        gain energy."""
        content = write(make_problem(), make_mesh(), tmp_path)

        assert "epsR[Diel_substrate] = Complex[4.4, -0.0044];" in content

    def test_lossless_dielectric_has_zero_imaginary_part(self, tmp_path):
        problem = make_problem(
            solids={
                "substrate": SolidSpec("substrate", "dielectric", Dielectric(2.2, 0.0)),
                "port_1": SolidSpec("port_1", "port"),
            }
        )

        content = write(problem, make_mesh(), tmp_path)

        assert "epsR[Diel_substrate] = Complex[2.2, -0.0];" in content

    def test_magnetic_material_emits_a_permeability_override(self, tmp_path):
        problem = make_problem(
            solids={
                "substrate": SolidSpec(
                    "substrate", "dielectric", Dielectric(4.4, 0.001, mu_r=2.5)
                ),
                "port_1": SolidSpec("port_1", "port"),
            }
        )

        content = write(problem, make_mesh(), tmp_path)

        assert "muR[Diel_substrate]" in content

    def test_non_magnetic_material_omits_the_override(self, tmp_path):
        content = write(make_problem(), make_mesh(), tmp_path)

        assert "muR[Diel_substrate]" not in content
        assert "muR[] = 1.;" in content

    def test_port_direction_reaches_the_excitation_vector(self, tmp_path):
        mesh = make_mesh(
            port_regions={
                1: PortMesh(
                    number=1,
                    region=port_region(1),
                    direction="x",
                    z0=50.0,
                    gap=1.6e-3,
                    width=3.0e-3,
                    center=(0.0, 0.0, 0.0008),
                )
            }
        )

        content = write(make_problem(), mesh, tmp_path)

        assert "dir_1[] = Vector[1., 0., 0.];" in content

    def test_multiple_dielectrics_are_all_emitted(self, tmp_path):
        problem = make_problem(
            solids={
                "substrate": SolidSpec(
                    "substrate", "dielectric", Dielectric(4.4, 0.001)
                ),
                "superstrate": SolidSpec(
                    "superstrate", "dielectric", Dielectric(2.2, 0.0)
                ),
                "port_1": SolidSpec("port_1", "port"),
            }
        )
        mesh = make_mesh(
            dielectric_regions={
                "substrate": dielectric_region(0),
                "superstrate": dielectric_region(1),
            }
        )

        content = write(problem, mesh, tmp_path)

        assert "Diel_substrate = Region[100];" in content
        assert "Diel_superstrate = Region[101];" in content
        assert "Domain = Region[{Diel_substrate, Diel_superstrate, Air}];" in content

    def test_dielectrics_are_emitted_in_sorted_order(self, tmp_path):
        """Sorted iteration is what makes the output reproducible across runs;
        dict ordering would otherwise leak in."""
        problem = make_problem(
            solids={
                "zsub": SolidSpec("zsub", "dielectric", Dielectric(4.4)),
                "asub": SolidSpec("asub", "dielectric", Dielectric(2.2)),
                "port_1": SolidSpec("port_1", "port"),
            }
        )
        mesh = make_mesh(
            dielectric_regions={
                "zsub": dielectric_region(1),
                "asub": dielectric_region(0),
            }
        )

        content = write(problem, mesh, tmp_path)

        assert content.index("Diel_asub =") < content.index("Diel_zsub =")


# ---------------------------------------------------------------------
# Multi-port
# ---------------------------------------------------------------------
class TestTwoPort:
    @pytest.fixture
    def two_port(self):
        problem = make_problem(
            solids={
                "substrate": SolidSpec(
                    "substrate", "dielectric", Dielectric(4.4, 0.001)
                ),
                "line": SolidSpec("line", "pec"),
                "port_1": SolidSpec("port_1", "port"),
                "port_2": SolidSpec("port_2", "port"),
            },
            ports=[PortSpec("port_1", 1, 50.0, "z"), PortSpec("port_2", 2, 50.0, "z")],
        )
        mesh = make_mesh(
            port_regions={
                number: PortMesh(
                    number=number,
                    region=port_region(number),
                    direction="z",
                    z0=50.0,
                    gap=1.6e-3,
                    width=3.0e-3,
                    center=(0.0, 0.0, 0.0008),
                )
                for number in (1, 2)
            }
        )
        return problem, mesh

    def test_both_ports_are_declared(self, two_port, tmp_path):
        problem, mesh = two_port

        content = write(problem, mesh, tmp_path)

        assert f"Port_1 = Region[{port_region(1)}];" in content
        assert f"Port_2 = Region[{port_region(2)}];" in content
        assert "Ports  = Region[{Port_1, Port_2}];" in content

    def test_second_port_reuses_the_factorisation(self, two_port, tmp_path):
        """The matrix is assembled and factorised once; every port after the
        first only rebuilds the right-hand side. Losing that turns an N-port
        sweep into N full solves.

        Only the ``Analysis`` resolution is checked -- ``AnalysisSinglePort``
        is a separate entry point that always does one full solve.
        """
        problem, mesh = two_port

        analysis = resolution_block(write(problem, mesh, tmp_path), "Analysis")

        assert analysis.count("Generate[A] ; Solve[A] ;") == 1
        assert analysis.count("GenerateRightHandSideGroup[A, Ports] ;") == 1
        assert analysis.count("SolveAgain[A] ;") == 1

    def test_one_port_needs_no_right_hand_side_rebuild(self, tmp_path):
        analysis = resolution_block(
            write(make_problem(), make_mesh(), tmp_path), "Analysis"
        )

        assert analysis.count("Generate[A] ; Solve[A] ;") == 1
        assert "SolveAgain[A] ;" not in analysis

    def test_each_port_is_driven_in_turn(self, two_port, tmp_path):
        problem, mesh = two_port

        content = write(problem, mesh, tmp_path)

        assert "$ActivePort = 1" in content
        assert "$ActivePort = 2" in content

    def test_s_parameters_are_written_once_per_port(self, two_port, tmp_path):
        problem, mesh = two_port

        content = write(problem, mesh, tmp_path)

        assert content.count("PostOperation[Get_SParameters] ;") == 2


# ---------------------------------------------------------------------
# Boundary conditions and element order
# ---------------------------------------------------------------------
class TestOptions:
    def test_silver_muller_emits_an_absorbing_boundary(self, tmp_path):
        content = write(make_problem(), make_mesh(), tmp_path)

        assert "Abc" in content
        assert "Pml" not in content

    def test_pml_boundary_emits_a_pml_region(self, tmp_path):
        mesh = make_mesh(
            boundary="pml",
            pml_region=210,
            inner_bbox=(-0.07, -0.07, -0.02, 0.07, 0.07, 0.02),
            pml_thick=0.01,
        )

        content = write(make_problem(), mesh, tmp_path)

        assert "Region[210]" in content

    @pytest.mark.parametrize("order", [1, 2])
    def test_element_order_is_written_as_the_feorder_constant(self, tmp_path, order):
        """The second-order basis functions are always present in the file but
        gated behind ``If (FEorder == 2)``, so this single constant is what
        actually selects the element order."""
        content = write(
            make_problem(options=FEMOptions(fe_order=order)), make_mesh(), tmp_path
        )

        assert f"FEorder = {order}" in content

    def test_the_gated_second_order_basis_function_is_present(self, tmp_path):
        content = write(
            make_problem(options=FEMOptions(fe_order=2)), make_mesh(), tmp_path
        )

        assert "If (FEorder == 2)" in content
        assert "BF_Edge_2E" in content

    def test_element_order_is_the_only_difference_between_orders(self, tmp_path):
        first = write(
            make_problem(options=FEMOptions(fe_order=1)), make_mesh(), tmp_path
        )
        second = write(
            make_problem(options=FEMOptions(fe_order=2)), make_mesh(), tmp_path
        )

        assert first != second
        assert first.replace("FEorder = 1", "FEorder = 2") == second

    def test_symmetry_plane_emits_its_region(self, tmp_path):
        mesh = make_mesh(sym_region=600, sym_kind="pmc", sym_axis=0, sym_plane=0.0)

        content = write(make_problem(), mesh, tmp_path)

        assert "600" in content

    def test_lossy_conductor_emits_a_surface_impedance(self, tmp_path):
        mesh = make_mesh(impedance_regions=[(350, 5.8e7)])

        content = write(make_problem(), mesh, tmp_path)

        assert "350" in content
