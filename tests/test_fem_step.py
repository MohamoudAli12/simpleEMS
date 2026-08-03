"""Tests for the standalone STEP-to-FEM entry point.

``simulate_step_FEM`` needs no CSXCAD geometry: it reads the solids straight
out of a STEP file, decides what each one is, meshes it and solves it. The
deciding is the interesting part, because it happens two ways -- explicit
overrides, and guessing from the solid's name -- and a solid that lands in the
wrong role still meshes and still solves. A substrate mistaken for a conductor
produces a full set of plausible S-parameters for a structure that does not
exist.

``_problem_from_step`` is therefore pinned in detail here, and the mesh and
solve stages get one test each at the bottom.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.needs_cadquery, pytest.mark.needs_csxcad]

pytest.importorskip("cadquery")

# The STEP path itself needs no CSXCAD geometry, but fem_backend imports the
# bindings at module scope for the paths that do, so this module cannot even be
# collected without them.
pytest.importorskip("CSXCAD")
pytest.importorskip("openEMS")

# gmsh dlopen()s libGLU at import time, so on a host without it the import
# raises OSError rather than ImportError. importorskip only catches the latter,
# which turns a missing system library into a collection error for the whole
# module; skip on any import failure instead.
try:
    import gmsh  # noqa: F401
except Exception as error:  # pragma: no cover - depends on the host
    pytest.skip(f"gmsh is not importable: {error}", allow_module_level=True)

import cadquery as cq  # noqa: E402

from simpleEMS.fem_backend import (  # noqa: E402
    FEMOptions,
    _problem_from_step,
    simulate_step_FEM,
)


def box_at(x, y, z, w, h, d):
    """A box of the given size with its minimum corner at ``(x, y, z)``."""
    return cq.Solid.makeBox(w, h, d).locate(cq.Location(cq.Vector(x, y, z)))


@pytest.fixture(scope="module")
def named_step(tmp_path_factory):
    """A STEP file whose solid names cover every guessable role and one that
    is not guessable at all."""
    asm = cq.Assembly(name="model")
    asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
    asm.add(box_at(-10, -10, -1.7, 20, 20, 0.1), name="ground")
    asm.add(box_at(-1.5, -8, 0, 3, 16, 0.1), name="trace")
    asm.add(box_at(-1.5, -8, -1.6, 3, 0.1, 1.6), name="port_1")
    asm.add(box_at(8, 8, 2, 1, 1, 1), name="widget")
    path = tmp_path_factory.mktemp("fem_step") / "named.step"
    asm.save(str(path))
    return path


def build(named_step, **overrides):
    kwargs = dict(
        step_file=str(named_step),
        freqs=np.linspace(2e9, 3e9, 11),
        dielectrics=None,
        pec=None,
        lossy_conductor=None,
        ports=None,
        charac_imp=50.0,
        FEM_options=None,
    )
    kwargs.update(overrides)
    return _problem_from_step(**kwargs)


def role_of(problem, name):
    return problem.solids[name].role


# ---------------------------------------------------------------------
# _problem_from_step -- role assignment by name
# ---------------------------------------------------------------------
class TestRoleGuessing:
    def test_every_solid_is_accounted_for(self, named_step):
        prob = build(named_step)

        assert set(prob.solids) == {
            "substrate",
            "ground",
            "trace",
            "port_1",
            "widget",
        }

    @pytest.mark.parametrize(
        ("name", "role"),
        [
            ("substrate", "dielectric"),
            ("ground", "pec"),
            ("trace", "pec"),
            ("port_1", "port"),
        ],
    )
    def test_names_are_read_as_roles(self, named_step, name, role):
        prob = build(named_step)

        assert role_of(prob, name) == role

    def test_an_unrecognised_name_is_ignored(self, named_step):
        """Left out of the simulation rather than guessed at."""
        prob = build(named_step)

        assert role_of(prob, "widget") == "ignore"

    def test_a_guessed_dielectric_gets_a_placeholder_permittivity(self, named_step):
        """eps_r = 1 is deliberately wrong-but-harmless: the name says it is a
        dielectric but nothing says which one, so it has to be overridden."""
        prob = build(named_step)

        assert prob.solids["substrate"].dielectric is not None
        assert prob.solids["substrate"].dielectric.eps_r == 1.0

    def test_a_guessed_port_is_registered(self, named_step):
        prob = build(named_step)

        assert [p.solid for p in prob.ports] == ["port_1"]

    def test_a_guessed_port_takes_the_default_impedance(self, named_step):
        prob = build(named_step, charac_imp=75.0)

        assert prob.ports[0].z0 == pytest.approx(75.0)


# ---------------------------------------------------------------------
# _problem_from_step -- explicit overrides
# ---------------------------------------------------------------------
class TestExplicitRoles:
    def test_dielectrics_override_the_guess(self, named_step):
        prob = build(named_step, dielectrics={"trace": (4.4, 0.001)})

        assert role_of(prob, "trace") == "dielectric"

    def test_the_declared_permittivity_is_used(self, named_step):
        prob = build(named_step, dielectrics={"substrate": (4.4, 0.02)})

        material = prob.solids["substrate"].dielectric
        assert material.eps_r == pytest.approx(4.4)
        assert material.tan_d == pytest.approx(0.02)

    def test_pec_overrides_the_guess(self, named_step):
        prob = build(named_step, pec=["substrate"])

        assert role_of(prob, "substrate") == "pec"

    def test_a_lossy_conductor_carries_its_conductivity(self, named_step):
        prob = build(named_step, lossy_conductor={"trace": 5.8e7})

        assert role_of(prob, "trace") == "lossy_conductor"
        assert prob.solids["trace"].sigma == pytest.approx(5.8e7)

    def test_an_unguessable_solid_can_be_named_explicitly(self, named_step):
        prob = build(named_step, pec=["widget"])

        assert role_of(prob, "widget") == "pec"

    def test_an_explicit_port_can_be_added(self, named_step):
        prob = build(named_step, ports={"widget": {}})

        assert role_of(prob, "widget") == "port"

    def test_explicit_ports_add_to_the_guessed_ones(self, named_step):
        """Naming one port explicitly does not switch guessing off for the
        others, so a file with a ``port_``-named solid ends up with both. To
        get only the explicit one, take the other out of the guess by naming
        it too."""
        prob = build(named_step, ports={"widget": {}})

        assert sorted(p.solid for p in prob.ports) == ["port_1", "widget"]

    def test_naming_the_guessed_solid_something_else_drops_it(self, named_step):
        prob = build(named_step, ports={"widget": {}}, pec=["port_1"])

        assert [p.solid for p in prob.ports] == ["widget"]

    def test_port_settings_are_honoured(self, named_step):
        prob = build(
            named_step,
            ports={"port_1": {"number": 3, "z0": 75.0, "direction": "x"}},
        )

        port = prob.ports[0]
        assert port.number == 3
        assert port.z0 == pytest.approx(75.0)
        assert port.direction == "x"

    def test_port_defaults_fill_in(self, named_step):
        prob = build(named_step, ports={"port_1": {}}, charac_imp=60.0)

        port = prob.ports[0]
        assert port.number == 1
        assert port.z0 == pytest.approx(60.0)
        assert port.direction == "z"

    def test_dielectrics_win_over_pec_for_the_same_solid(self, named_step):
        """The branch order decides; pinned so a reordering is caught."""
        prob = build(named_step, dielectrics={"trace": (4.4, 0.001)}, pec=["trace"])

        assert role_of(prob, "trace") == "dielectric"

    def test_lossy_conductor_wins_over_pec(self, named_step):
        prob = build(named_step, lossy_conductor={"trace": 5.8e7}, pec=["trace"])

        assert role_of(prob, "trace") == "lossy_conductor"

    def test_pec_wins_over_an_explicit_port(self, named_step):
        prob = build(named_step, pec=["port_1"], ports={"port_1": {}})

        assert role_of(prob, "port_1") == "pec"
        assert prob.ports == []


# ---------------------------------------------------------------------
# _problem_from_step -- ports and options
# ---------------------------------------------------------------------
class TestProblemAssembly:
    @pytest.fixture(scope="class")
    def two_port_step(self, tmp_path_factory):
        asm = cq.Assembly(name="model")
        asm.add(box_at(-10, -10, -1.6, 20, 20, 1.6), name="substrate")
        asm.add(box_at(-10, -10, -1.7, 20, 20, 0.1), name="ground")
        asm.add(box_at(-1.5, -8, 0, 3, 16, 0.1), name="trace")
        asm.add(box_at(-1.5, -8, -1.6, 3, 0.1, 1.6), name="port_a")
        asm.add(box_at(-1.5, 7.9, -1.6, 3, 0.1, 1.6), name="port_b")
        path = tmp_path_factory.mktemp("fem_step2") / "twoport.step"
        asm.save(str(path))
        return path

    def test_ports_come_out_sorted_by_number(self, two_port_step):
        """The S-matrix row/column order follows this list, so a dict written
        out of order must not reorder the matrix."""
        prob = _problem_from_step(
            str(two_port_step),
            np.linspace(2e9, 3e9, 11),
            None,
            None,
            None,
            {"port_a": {"number": 2}, "port_b": {"number": 1}},
            50.0,
            None,
        )

        assert [p.number for p in prob.ports] == [1, 2]
        assert [p.solid for p in prob.ports] == ["port_b", "port_a"]

    def test_a_repeated_port_number_is_registered_once(self, two_port_step):
        """Two solids claiming port 1 would otherwise produce two rows for the
        same port and desync the S-matrix."""
        prob = _problem_from_step(
            str(two_port_step),
            np.linspace(2e9, 3e9, 11),
            None,
            None,
            None,
            {"port_a": {"number": 1}, "port_b": {"number": 1}},
            50.0,
            None,
        )

        assert len(prob.ports) == 1

    def test_auto_numbered_ports_are_sequential(self, two_port_step):
        prob = _problem_from_step(
            str(two_port_step),
            np.linspace(2e9, 3e9, 11),
            None,
            None,
            None,
            None,
            50.0,
            None,
        )

        assert sorted(p.number for p in prob.ports) == [1, 2]

    def test_fem_options_are_applied(self, named_step):
        options = FEMOptions(fe_order=2, num_solve_points=7)

        prob = build(named_step, FEM_options=options)

        assert prob.fe_order == 2
        assert prob.num_solve_points == 7

    def test_omitting_options_leaves_the_defaults(self, named_step):
        prob = build(named_step)

        assert prob.fe_order == FEMOptions().fe_order

    def test_frequencies_are_stored_as_floats(self, named_step):
        prob = build(named_step, freqs=[2e9, 3e9])

        assert prob.freqs.dtype == np.dtype(float)

    def test_the_step_path_is_expanded(self, named_step):
        prob = build(named_step)

        assert prob.step_file == str(named_step)

    def test_the_problem_is_named_for_its_output_files(self, named_step):
        prob = build(named_step)

        assert prob.name == "structure"


# ---------------------------------------------------------------------
# simulate_step_FEM
# ---------------------------------------------------------------------
class TestSimulateStepFEM:
    def test_a_portless_geometry_is_rejected(self, tmp_path):
        """Reached before any meshing, so it costs nothing to hit."""
        asm = cq.Assembly(name="model")
        asm.add(box_at(0, 0, 0, 10, 10, 1), name="widget")
        step = tmp_path / "portless.step"
        asm.save(str(step))

        with pytest.raises(RuntimeError, match="No ports found or specified"):
            simulate_step_FEM(
                str(step),
                np.linspace(2e9, 3e9, 11),
                output_path=tmp_path / "out",
                run=False,
            )

    def test_the_error_says_how_to_fix_it(self, tmp_path):
        asm = cq.Assembly(name="model")
        asm.add(box_at(0, 0, 0, 10, 10, 1), name="widget")
        step = tmp_path / "portless.step"
        asm.save(str(step))

        with pytest.raises(RuntimeError, match=r"pass `ports=\.\.\.`"):
            simulate_step_FEM(
                str(step),
                np.linspace(2e9, 3e9, 11),
                output_path=tmp_path / "out",
                run=False,
            )


@pytest.mark.slow
class TestMeshing:
    """The mesh stage, without solving.

    Meshing is where the symmetry cut happens, and a mis-cut model solves
    perfectly well while representing half the wrong structure.
    """

    def mesh_only(self, named_step, out, **kwargs):
        """Mesh and read back the metadata, without running the solver.

        ``run=False`` meshes and then still tries to read results, so against
        a fresh directory it always ends in "FEM results not found". Reaching
        that error *is* the successful mesh -- anything else means meshing
        itself failed.
        """
        import json

        with pytest.raises(RuntimeError, match="FEM results not found"):
            simulate_step_FEM(
                str(named_step),
                np.linspace(2e9, 3e9, 5),
                dielectrics={"substrate": (4.4, 0.001)},
                output_path=out,
                run=False,
                verbose=False,
                FEM_num_solve_points=4,
                FEM_elems_per_wavelength=4.0,
                FEM_min_layers=1,
                **kwargs,
            )
        return json.loads((out / "fem_mesh.json").read_text())

    def test_meshing_writes_its_metadata(self, named_step, tmp_path):
        meta = self.mesh_only(named_step, tmp_path / "out")

        assert meta["msh_path"]
        assert meta["pro_path"]

    def test_the_mesh_and_problem_files_exist(self, named_step, tmp_path):
        from pathlib import Path

        meta = self.mesh_only(named_step, tmp_path / "out")

        assert Path(meta["msh_path"]).is_file()
        assert Path(meta["pro_path"]).is_file()

    def test_the_port_is_recorded(self, named_step, tmp_path):
        meta = self.mesh_only(named_step, tmp_path / "out")

        assert meta["port_numbers"] == [1]
        assert meta["ref_impedances"]["1"] == pytest.approx(50.0)

    def test_no_symmetry_is_recorded_by_default(self, named_step, tmp_path):
        meta = self.mesh_only(named_step, tmp_path / "out")

        assert meta["symmetry_axis"] is None
        assert meta["symmetry_plane"] is None

    def test_a_symmetry_plane_is_recorded(self, named_step, tmp_path):
        """The far-field transform needs all three to mirror the half model
        back to a whole one; it refuses to guess if any are missing."""
        meta = self.mesh_only(
            named_step, tmp_path / "sym", FEM_symmetry=("x", "pmc", 0.0)
        )

        assert meta["symmetry_axis"] == 0
        assert meta["symmetry_plane"] == pytest.approx(0.0)
        assert meta["symmetry_kind"] == "pmc"

    def nodes(self, msh_path):
        """Node coordinates of a written mesh, as an ``(n, 3)`` array."""
        import gmsh

        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.open(str(msh_path))
            _tags, coords, _param = gmsh.model.mesh.getNodes()
            return np.asarray(coords, dtype=float).reshape(-1, 3)
        finally:
            gmsh.finalize()

    def test_the_cut_removes_everything_below_the_plane(self, named_step, tmp_path):
        """A cut that leaves nodes on the far side means the half model still
        carries part of the mirrored structure, and the far-field mirroring
        would then double-count it."""
        meta = self.mesh_only(
            named_step, tmp_path / "half", FEM_symmetry=("x", "pmc", 0.0)
        )

        x = self.nodes(meta["msh_path"])[:, 0]

        assert x.min() >= -1e-9

    def test_the_uncut_mesh_spans_both_sides(self, named_step, tmp_path):
        """Control for the test above."""
        meta = self.mesh_only(named_step, tmp_path / "whole")

        x = self.nodes(meta["msh_path"])[:, 0]

        assert x.min() < -1e-3

    def test_the_cut_shrinks_the_mesh(self, named_step, tmp_path):
        """The whole point of the symmetry option: half the mesh, half the
        solve."""
        whole = self.mesh_only(named_step, tmp_path / "whole")
        half = self.mesh_only(
            named_step, tmp_path / "half", FEM_symmetry=("x", "pmc", 0.0)
        )

        assert len(self.nodes(half["msh_path"])) < len(self.nodes(whole["msh_path"]))

    def test_the_structure_bbox_survives_the_cut(self, named_step, tmp_path):
        """``bbox`` describes the structure, not the meshed region, so cutting
        the domain must not move it."""
        whole = self.mesh_only(named_step, tmp_path / "whole")
        half = self.mesh_only(
            named_step, tmp_path / "half", FEM_symmetry=("x", "pmc", 0.0)
        )

        assert half["bbox"] == pytest.approx(whole["bbox"])

    def test_domain_bbox_is_recorded_before_the_cut(self, named_step, tmp_path):
        """Pinned because ``fem_radiation.compute_pattern`` depends on it: it
        clamps the Huygens box to the symmetry plane by hand precisely because
        this value still describes the uncut domain. If the cut ever started
        updating it, that clamp would become a double correction."""
        whole = self.mesh_only(named_step, tmp_path / "whole")
        half = self.mesh_only(
            named_step, tmp_path / "half", FEM_symmetry=("x", "pmc", 0.0)
        )

        assert half["domain_bbox"] == pytest.approx(whole["domain_bbox"])


@pytest.mark.slow
@pytest.mark.needs_getdp_bin
def test_a_step_file_solves_end_to_end(named_step, tmp_path):
    """Badly converged on purpose -- see the note in test_solver_smoke.py.
    What this proves is that a STEP file with no CSXCAD geometry behind it
    goes all the way to a results bundle the rest of the pipeline accepts."""
    from simpleEMS.sim_tools import SimData

    freqs = np.linspace(2e9, 3e9, 5)

    data = simulate_step_FEM(
        str(named_step),
        freqs,
        dielectrics={"substrate": (4.4, 0.001)},
        output_path=tmp_path / "out",
        verbose=False,
        FEM_num_solve_points=4,
        FEM_elems_per_wavelength=4.0,
        FEM_min_layers=1,
    )

    assert isinstance(data, SimData)
    assert data.freqs == pytest.approx(freqs)
    assert data.s11.shape == freqs.shape
    assert np.all(np.isfinite(data.s11))
    assert np.all(np.abs(data.s11) <= 1.0 + 1e-6)
    assert np.all(data.vswr >= 1.0)
    assert data.ref_impedance == pytest.approx(50.0)
