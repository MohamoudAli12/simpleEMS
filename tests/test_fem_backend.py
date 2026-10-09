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

from simpleEMS import fem_backend  # noqa: E402
from simpleEMS.fem_backend import FEMOptions, PortSpec, Problem  # noqa: E402


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


# ---------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------
class TestAirPadFaces:
    def test_one_value_pads_every_face(self):
        assert fem_backend._air_pad_faces_mm(2) == ((2.0, 2.0),) * 3

    def test_three_values_pad_each_axis_on_both_sides(self):
        assert fem_backend._air_pad_faces_mm([1, 2, 3]) == (
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
        )

    def test_three_pairs_pad_each_face(self):
        faces = fem_backend._air_pad_faces_mm([(1, 2), (3, 4), (5, 6)])

        assert faces == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    @pytest.mark.parametrize(
        "air_pad_mm",
        ["wide", [1, 2], [(1, 2), (3, 4), 5], object()],
        ids=["text", "two-values", "mixed", "not-a-sequence"],
    )
    def test_anything_else_is_refused(self, air_pad_mm):
        with pytest.raises(ValueError, match="air_pad_mm must be one value"):
            fem_backend._air_pad_faces_mm(air_pad_mm)


class TestFEMOptionsValidation:
    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"fe_order": 3}, "fe_order must be 1 or 2"),
            ({"num_solve_points": 3}, "num_solve_points must be >= 4"),
            (
                {"num_solve_points": 10, "max_solve_points": 6},
                "max_solve_points must be >= num_solve_points",
            ),
            (
                {"air_pad_mm": [(1, -1), (1, 1), (1, 1)]},
                "air_pad_mm cannot be negative",
            ),
        ],
        ids=["fe-order", "too-few-solves", "cap-below-budget", "negative-pad"],
    )
    def test_a_bad_option_is_refused(self, options, message):
        with pytest.raises(ValueError, match=message):
            FEMOptions(**options)

    def test_a_zero_pad_is_allowed(self):
        assert FEMOptions(air_pad_mm=0).air_pad_faces_mm == ((0.0, 0.0),) * 3


class TestPortSpecValidation:
    def test_the_solid_list_defaults_to_the_port_solid(self):
        assert PortSpec("port_1", 1).solids == ["port_1"]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"kind": "coax"}, "kind must be 'lumped' or 'wave'"),
            ({"prop_dir": "w"}, "prop_dir must be 'x', 'y' or 'z'"),
            ({"kind": "wave", "direction": "y", "prop_dir": "y"}, "is the axis"),
        ],
        ids=["kind", "prop-dir", "along-the-field"],
    )
    def test_a_bad_port_is_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PortSpec("port_1", 1, **kwargs)


class TestProblemOptions:
    @pytest.mark.parametrize(
        ("name", "value"),
        [("air_pad_mm", 3.0), ("port_type", "waveport")],
    )
    def test_reads_the_option_through(self, name, value):
        problem = Problem(
            step_file="s.step",
            freqs=np.array([1e9]),
            options=FEMOptions(**{name: value}),
        )

        assert getattr(problem, name) == value


# ---------------------------------------------------------------------
# Roles, fingerprint, mesh reuse
# ---------------------------------------------------------------------
class TestRolesAndFingerprint:
    def test_a_conducting_sheet_is_a_lossy_conductor(self):
        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        csx.AddConductingSheet("trace", conductivity=5.8e7, thickness=35e-6)

        roles, _dielectrics, _ports, sigmas = fem_backend._csx_roles(csx, 1e9)

        assert roles["trace"] == "lossy_conductor"
        assert sigmas["trace"] == pytest.approx(5.8e7)

    def test_an_unreadable_module_is_hashed_by_name(self, tmp_path):
        class Module:
            __file__ = str(tmp_path / "gone.py")

            def __str__(self):
                return "module-name"

        assert fem_backend._module_bytes(Module()) == b"module-name"

    @staticmethod
    def fingerprint(build):
        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        build(csx.AddMetal("metal"))
        return fem_backend._mesh_fingerprint(csx, np.array([1e9]), None)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (
                lambda m: m.AddLinPoly([[0, 1, 1], [0, 0, 1]], "z", 0, 0.035),
                lambda m: m.AddLinPoly([[0, 2, 2], [0, 0, 1]], "z", 0, 0.035),
            ),
            (
                lambda m: m.AddCylinder([0, 0, 0], [0, 0, 1], 0.3),
                lambda m: m.AddCylinder([1, 0, 0], [1, 0, 1], 0.3),
            ),
            (
                lambda m: m.AddCylindricalShell([0, 0, 0], [0, 0, 1], 0.5, 0.2),
                lambda m: m.AddCylindricalShell([0, 0, 0], [0, 0, 1], 0.5, 0.3),
            ),
        ],
        ids=["polygon", "via", "antipad"],
    )
    def test_moving_or_resizing_a_shape_changes_the_fingerprint(self, first, second):
        assert self.fingerprint(first) != self.fingerprint(second)

    def test_other_shapes_count_by_type(self):
        sphere = self.fingerprint(lambda m: m.AddSphere([0, 0, 0], 1.0))

        assert sphere == self.fingerprint(lambda m: m.AddSphere([5, 5, 5], 2.0))
        assert sphere != self.fingerprint(lambda m: m.AddBox([0, 0, 0], [1, 1, 1]))

    def test_a_portless_problem_cannot_be_meshed(self, tmp_path):
        problem = Problem(step_file="s.step", freqs=np.array([1e9]))

        with pytest.raises(RuntimeError, match="No ports found"):
            fem_backend._mesh_problem(problem, tmp_path, verbose=False)

    def test_an_unchanged_geometry_reuses_its_mesh(self, tmp_path, capsys):
        import json

        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        csx.AddMetal("plate").AddBox([0, 0, 0], [1, 1, 1])
        freqs = np.array([1e9])
        msh = tmp_path / "structure.msh"
        msh.write_text("")
        (tmp_path / "fem_mesh.json").write_text(
            json.dumps(
                {
                    "fingerprint": fem_backend._mesh_fingerprint(csx, freqs, None),
                    "msh_path": str(msh),
                }
            )
        )

        assert fem_backend.build_mesh(csx, freqs, tmp_path) == str(msh)
        assert "reusing unchanged FEM mesh" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Re-referencing S to another impedance
# ---------------------------------------------------------------------
class TestRenormalise:
    def test_the_same_impedance_changes_nothing(self):
        s = np.array([[0.1 + 0.2j, 0.9], [0.9, 0.1]])

        assert fem_backend._renormalise(s, [50, 50], [50, 50]) is s

    def test_a_matched_load_reads_as_the_mismatch(self):
        """A 25 ohm line matched to itself, seen from 50 ohm, reflects
        (25 - 50) / (25 + 50) = -1/3."""
        s = np.array([[0.0 + 0j]])

        renormalised = fem_backend._renormalise(s, [25.0], [50.0])

        assert renormalised[0, 0] == pytest.approx(-1 / 3)

    @staticmethod
    def through_z(s, z_old, z_new):
        """Re-reference by way of the Z-matrix: the definition, not a shortcut."""
        identity = np.eye(len(z_old))
        root_old = np.diag(np.sqrt(z_old))
        z = root_old @ np.linalg.inv(identity - s) @ (identity + s) @ root_old
        inverse_root_new = np.diag(1 / np.sqrt(z_new))
        normalised = inverse_root_new @ z @ inverse_root_new
        return (normalised - identity) @ np.linalg.inv(normalised + identity)

    RECIPROCAL = np.array([[0.2 + 0.1j, 0.7 - 0.2j], [0.7 - 0.2j, 0.3j]])

    def test_the_same_change_on_every_port_matches_the_definition(self):
        renormalised = fem_backend._renormalise(self.RECIPROCAL, [35, 35], [50, 50])

        assert renormalised == pytest.approx(
            self.through_z(self.RECIPROCAL, [35, 35], [50, 50])
        )

    def test_reflections_match_the_definition_for_any_change(self):
        renormalised = fem_backend._renormalise(self.RECIPROCAL, [35, 60], [50, 50])

        expected = self.through_z(self.RECIPROCAL, [35, 60], [50, 50])
        assert np.diag(renormalised) == pytest.approx(np.diag(expected))

    def test_ports_changed_by_different_ratios_match_the_definition(self):
        renormalised = fem_backend._renormalise(self.RECIPROCAL, [35, 60], [50, 50])

        assert renormalised == pytest.approx(
            self.through_z(self.RECIPROCAL, [35, 60], [50, 50])
        )

    def test_a_reciprocal_network_stays_reciprocal(self):
        renormalised = fem_backend._renormalise(self.RECIPROCAL, [35, 60], [50, 50])

        assert renormalised[0, 1] == pytest.approx(renormalised[1, 0])


# ---------------------------------------------------------------------
# Building a problem from CSXCAD geometry
# ---------------------------------------------------------------------
@pytest.mark.needs_cadquery
class TestCoplanarPortPromotion:
    def test_a_two_gap_port_is_solved_as_a_wave_port(self, tmp_path, capsys):
        """openEMS draws a CPW port as one box per gap with 2 * Z0 on each; the
        FEM backend has to solve it as a wave port at Z0."""
        from CSXCAD import ContinuousStructure

        csx = ContinuousStructure()
        csx.AddMaterial("substrate", epsilon=4.4).AddBox([-5, -5, 0], [5, 5, 1.6])
        csx.AddMetal("trace").AddBox([-0.5, -5, 1.6], [0.5, 5, 1.635])
        port = csx.AddLumpedElement("port_resist_1", ny=0, caps=True, R=100)
        port.AddBox([-0.8, -5, 1.6], [-0.5, -4.9, 1.635])
        port.AddBox([0.5, -5, 1.6], [0.8, -4.9, 1.635])

        problem = fem_backend._build_problem(csx, np.array([1e9]), tmp_path)

        ((number, kind, z0),) = [(p.number, p.kind, p.z0) for p in problem.ports]
        assert (number, kind) == (1, "wave")
        assert z0 == pytest.approx(50.0)
        assert "coplanar waveguide" in capsys.readouterr().out
