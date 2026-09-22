"""Tests for :mod:`simpleEMS.fem_port_mode` -- the 2D wave-port mode solve.

Two tiers, mirroring the rest of the FEM suite. The first needs nothing but
numpy: the parsers, the mode picker and the view writer are plain functions
over arrays, and they are where the fiddly details live (which ``.res`` line
carries the eigenvalue, which pair of ``.pos`` steps is one complex mode, which
way round the local frame goes).

The second tier actually runs GetDP on cross-sections whose answers are known
in closed form -- a rectangular waveguide, a parallel-plate TEM line, and a
microstrip checked against :mod:`simpleEMS.calc`. That is the only tier that
can catch the formulation itself being wrong, so it is worth the solver.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simpleEMS import fem_port_mode
from simpleEMS.fem_port_mode import (
    PortModeSetup,
    _integrate_sq,
    _local_axes,
    _select_mode,
    _tri_areas,
    read_eigenvalues,
    read_pos_steps,
)

C0 = 299792458.0


def make_setup(**overrides) -> PortModeSetup:
    """A setup for a port face normal to y, in the xz plane."""
    defaults = {
        "number": 1,
        "pro_path": "structure_mode_1.pro",
        "msh_path": "structure_port_1.msh",
        "direction": "z",
        "prop_axis": 1,
        "plane_at": -0.015,
        "axes": (2, 0),
        "bounds": (0.0, -0.01, 0.0076, 0.01),
        "eps_max": 4.4,
        "z0": 50.0,
    }
    defaults.update(overrides)
    return PortModeSetup(**defaults)


class TestLocalAxes:
    @pytest.mark.parametrize(
        ("prop_axis", "expected"),
        [(0, (1, 2)), (1, (2, 0)), (2, (0, 1))],
        ids=["normal-x", "normal-y", "normal-z"],
    )
    def test_the_flattened_frame_is_right_handed(self, prop_axis, expected):
        assert _local_axes(prop_axis) == expected

    @pytest.mark.parametrize("prop_axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_local_x_cross_local_y_is_the_port_normal(self, prop_axis):
        # a left-handed frame would silently flip the sign of the mode
        u, v = _local_axes(prop_axis)
        eu, ev = np.eye(3)[u], np.eye(3)[v]
        assert np.allclose(np.cross(eu, ev), np.eye(3)[prop_axis])


class TestReadEigenvalues:
    def test_reads_the_value_out_of_each_solution_header(self, tmp_path):
        # GetDP puts the eigenvalue where a time-domain run puts the time
        res = tmp_path / "m.res"
        res.write_text(
            "$ResFormat /* GetDP */\n1.1 0\n$EndResFormat\n"
            "$Solution  /* DofData #0 */\n0 94.404739 -1.1e-16 0\n0.1 0.2\n"
            "$EndSolution\n"
            "$Solution  /* DofData #0 */\n0 7.8e-06 4.0e-06 1\n0.3 0.4\n"
            "$EndSolution\n"
        )
        vals = read_eigenvalues(res)
        assert len(vals) == 2
        assert vals[0] == pytest.approx(complex(94.404739, -1.1e-16))
        assert vals[1] == pytest.approx(complex(7.8e-06, 4.0e-06))

    def test_a_file_with_no_solutions_reads_as_empty(self, tmp_path):
        res = tmp_path / "m.res"
        res.write_text("$ResFormat\n1.1 0\n$EndResFormat\n")
        assert read_eigenvalues(res) == []


class TestReadPosSteps:
    def _view(self, nsteps: int, ntris: int = 2) -> str:
        out = ['View "et" {']
        for t in range(ntris):
            coords = ",".join(str(float(t + i)) for i in range(9))
            vals = ",".join(
                str(float(100 * s + t)) for s in range(nsteps) for _ in range(9)
            )
            out.append(f"VT({coords}){{{vals}}};")
        out.append("};")
        return "\n".join(out)

    def test_shapes_split_coordinates_from_per_step_values(self, tmp_path):
        pos = tmp_path / "et.pos"
        pos.write_text(self._view(nsteps=4, ntris=3))
        coords, steps = read_pos_steps(pos)
        assert coords.shape == (3, 3, 3)
        assert steps.shape == (3, 4, 3, 3)

    def test_steps_are_kept_in_order_so_a_mode_is_a_consecutive_pair(self, tmp_path):
        # GetDP writes a complex field as real step 2k then imaginary step 2k+1
        pos = tmp_path / "et.pos"
        pos.write_text(self._view(nsteps=6, ntris=1))
        _coords, steps = read_pos_steps(pos)
        assert steps[0, 0, 0, 0] == 0.0
        assert steps[0, 1, 0, 0] == 100.0
        assert steps[0, 5, 0, 0] == 500.0

    def test_a_mesh_based_view_is_rejected_with_a_pointed_message(self, tmp_path):
        # the failure this guards against is silent: the file exists and is
        # large, but holds no VT records at all
        pos = tmp_path / "et.pos"
        pos.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n")
        with pytest.raises(RuntimeError, match="GmshParsed"):
            read_pos_steps(pos)

    def test_an_empty_view_reads_as_empty_rather_than_raising(self, tmp_path):
        pos = tmp_path / "et.pos"
        pos.write_text('View "et" {\n};\n')
        coords, steps = read_pos_steps(pos)
        assert coords.size == 0


class TestSelectMode:
    def test_picks_the_largest_real_beta(self):
        k0 = 20.0
        betas = [3.0 + 0j, 36.0 + 0j, 12.0 + 0j]
        assert _select_mode(betas, k0, math.sqrt(4.4)) == 1

    def test_rejects_the_gradient_null_space_at_beta_near_zero(self):
        k0 = 20.0
        betas = [1e-6 + 0j, 2e-6 + 0j, 30.0 + 0j]
        assert _select_mode(betas, k0, math.sqrt(4.4)) == 2

    def test_rejects_evanescent_modes(self):
        # below cutoff beta^2 is negative, so getdp reports an imaginary beta
        k0 = 20.0
        betas = [0 + 40j, 25.0 + 0j]
        assert _select_mode(betas, k0, math.sqrt(4.4)) == 1

    def test_rejects_beta_above_the_densest_material(self):
        k0 = 20.0
        n_max = math.sqrt(4.4)  # beta cannot exceed ~41.9
        betas = [500.0 + 0j, 35.0 + 0j]
        assert _select_mode(betas, k0, n_max) == 1

    def test_no_guided_mode_raises_and_says_what_it_looked_for(self):
        with pytest.raises(RuntimeError, match="no guided mode"):
            _select_mode([1e-9 + 0j], 20.0, math.sqrt(4.4))

    # The numbers below are the measured spectrum of the conductor-backed CPW
    # cross-section in examples/CPWLine_2_45GHz.py at 2.45 GHz. It carries
    # three quasi-TEM modes, and the CPW mode -- the one the conformal-mapping
    # design targets at eps_eff 2.90 -- is the *third*, not the first.
    CPW_BETAS = [98.696 + 0j, 91.491 + 0j, 85.499 + 0j, 0j, 0.036 + 0j]
    CPW_K0 = 2 * math.pi * 2.45e9 / C0

    def test_a_multi_conductor_cross_section_needs_the_index(self):
        n_max = math.sqrt(4.4)
        picked = [_select_mode(self.CPW_BETAS, self.CPW_K0, n_max, i) for i in range(3)]
        assert picked == [0, 1, 2]
        eps = [(self.CPW_BETAS[i] / self.CPW_K0).real ** 2 for i in picked]
        # index 0 is the microstrip-like mode; index 2 is the CPW mode
        assert eps[0] == pytest.approx(3.69, abs=0.02)
        assert eps[2] == pytest.approx(2.77, abs=0.02)

    def test_modes_are_ranked_by_beta_not_by_solver_order(self):
        # Arpack does not return eigenpairs sorted, so the ranking has to be
        # imposed here or port_mode_index would mean nothing repeatable.
        shuffled = [85.499 + 0j, 98.696 + 0j, 91.491 + 0j]
        n_max = math.sqrt(4.4)
        assert _select_mode(shuffled, self.CPW_K0, n_max, 0) == 1
        assert _select_mode(shuffled, self.CPW_K0, n_max, 2) == 0

    def test_asking_for_a_mode_that_was_not_found_says_what_was(self):
        with pytest.raises(RuntimeError, match="only 3 guided mode"):
            _select_mode(self.CPW_BETAS, self.CPW_K0, math.sqrt(4.4), 5)

    def test_naming_a_mode_by_its_eps_eff_finds_it(self):
        # 2.90 is the conformal-mapping design value for that CPW line
        picked = _select_mode(self.CPW_BETAS, self.CPW_K0, math.sqrt(4.4), 0, 2.90)
        assert picked == 2

    def test_eps_eff_keeps_meaning_the_same_mode_when_modes_drop_out(self):
        # A coarser mesh resolved only the CPW mode; an index would then have
        # raised or, worse, pointed at a different mode. The target does not.
        thinned = [85.499 + 0j]
        assert _select_mode(thinned, self.CPW_K0, math.sqrt(4.4), 0, 2.90) == 0

    def test_eps_eff_takes_precedence_over_the_index(self):
        picked = _select_mode(self.CPW_BETAS, self.CPW_K0, math.sqrt(4.4), 0, 2.90)
        assert picked == 2  # index 0 would have been the microstrip-like mode

    @pytest.mark.needs_csxcad
    def test_an_eps_eff_below_one_is_rejected_up_front(self):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="cannot be below 1"):
            FEMOptions(port_mode_eps_eff=0.5)


class TestIntegration:
    def test_area_of_a_unit_triangle(self):
        coords = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
        assert _tri_areas(coords) == pytest.approx([0.5])

    def test_integrating_a_constant_field_gives_area_times_the_square(self):
        # exact for the product of two linear interpolants, so a constant is
        # the case that must come out on the nose
        coords = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
        values = np.ones((1, 3, 3), dtype=complex) * (3.0 + 4.0j)
        # |3+4j|^2 = 25, per component, three components -> 75, times area 0.5
        assert _integrate_sq(values, _tri_areas(coords)) == pytest.approx(37.5)


class TestWriteModeView:
    def test_the_view_is_written_back_into_3d_coordinates(self, tmp_path):
        setup = make_setup(prop_axis=1, axes=(2, 0), plane_at=-0.015)
        coords = np.array([[[0.001, 0.002, 0], [0.003, 0.004, 0], [0.005, 0.006, 0]]])
        values = np.zeros((1, 3, 3), dtype=complex)
        values[0, :, 0] = 1.0  # local x == 3D z
        values[0, :, 1] = 2.0  # local y == 3D x
        out = tmp_path / "mode_1.pos"
        fem_port_mode._write_mode_view(out, coords, values, setup, "mode_1")
        text = out.read_text()

        assert text.startswith('View "mode_1" {')
        assert "TIME{0,1};" in text
        coords_out, steps = read_pos_steps(out)
        # every node sits on the port plane, along the propagation axis
        assert np.allclose(coords_out[:, :, 1], -0.015)
        # local x -> 3D z, local y -> 3D x, nothing along the normal
        assert np.allclose(steps[0, 0, :, 2], 1.0)
        assert np.allclose(steps[0, 0, :, 0], 2.0)
        assert np.allclose(steps[0, 0, :, 1], 0.0)

    def test_the_imaginary_part_becomes_the_second_step(self, tmp_path):
        setup = make_setup()
        coords = np.zeros((1, 3, 3))
        values = np.full((1, 3, 3), 1.0 + 7.0j, dtype=complex)
        out = tmp_path / "mode_1.pos"
        fem_port_mode._write_mode_view(out, coords, values, setup, "mode_1")
        _c, steps = read_pos_steps(out)
        assert steps.shape[1] == 2
        assert np.allclose(steps[0, 1, :, 2], 7.0)  # local x -> 3D z


class TestSetupRoundTrip:
    def test_survives_the_trip_through_fem_mesh_json(self):
        setup = make_setup()
        again = PortModeSetup.from_dict(setup.to_dict())
        assert again == setup

    def test_to_dict_is_json_serialisable(self):
        import json

        json.dumps(make_setup().to_dict())


class TestEigenPar:
    def test_written_so_arpack_never_stops_to_ask(self, tmp_path):
        # an inherited stdin that never delivers a line hangs the whole sweep
        fem_port_mode._write_eigen_par(tmp_path)
        par = tmp_path / "eigen.par"
        assert par.exists()
        assert par.read_text().splitlines()[:3] == ["0.0001", "0", "50"]

    def test_an_existing_file_is_left_alone(self, tmp_path):
        par = tmp_path / "eigen.par"
        par.write_text("1e-8\n1\n80\n")
        fem_port_mode._write_eigen_par(tmp_path)
        assert par.read_text() == "1e-8\n1\n80\n"


# ----------------------------------------------------------------------------
# Solver tier: cross-sections whose propagation constant is known in closed
# form. This is the only tier that can catch the eigenformulation itself being
# wrong -- the block split between the stiffness and eigenvalue matrices, the
# sign that decides whether GetDP reports beta or j*beta, and the space pair.
# ----------------------------------------------------------------------------
pytest.importorskip("gmsh")

import gmsh  # noqa: E402

from simpleEMS.fem_geometry import Mesh, PortMesh  # noqa: E402
from simpleEMS.fem_materials import (  # noqa: E402
    ABC,
    AIR,
    PEC,
    Dielectric,
    dielectric_region,
    port_region,
)


def _rect_section(path, w, h, eps_r=None, walls="all", split_at=None, lc=None):
    """Mesh a rectangular cross-section with the region tags the .pro expects.

    ``walls`` is ``"all"`` for a fully enclosed guide or ``"plates"`` for PEC
    only along the top and bottom, leaving the sides as the natural (magnetic
    wall) condition. ``split_at`` puts a dielectric interface at that height,
    with a PEC trace of that width centred on it.
    """
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("section")
        g = gmsh.model.geo
        lc = lc or min(w, h) / 12

        if split_at is None:
            pts = [
                g.addPoint(0, 0, 0, lc),
                g.addPoint(w, 0, 0, lc),
                g.addPoint(w, h, 0, lc),
                g.addPoint(0, h, 0, lc),
            ]
            ls = [g.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            surf = g.addPlaneSurface([g.addCurveLoop(ls)])
            g.synchronize()
            gmsh.model.addPhysicalGroup(2, [surf], AIR)
            keep = ls if walls == "all" else [ls[0], ls[2]]
            gmsh.model.addPhysicalGroup(1, keep, PEC)
        else:
            hs, trace_w = split_at
            xc = w / 2
            lcf = trace_w / 10
            p = [
                g.addPoint(0, 0, 0, lc),
                g.addPoint(w, 0, 0, lc),
                g.addPoint(w, hs, 0, lc),
                g.addPoint(xc + trace_w / 2, hs, 0, lcf),
                g.addPoint(xc - trace_w / 2, hs, 0, lcf),
                g.addPoint(0, hs, 0, lc),
                g.addPoint(w, h, 0, lc),
                g.addPoint(0, h, 0, lc),
            ]
            bot = g.addLine(p[0], p[1])
            rs = g.addLine(p[1], p[2])
            i_r = g.addLine(p[2], p[3])
            trace = g.addLine(p[3], p[4])
            i_l = g.addLine(p[4], p[5])
            ls_ = g.addLine(p[5], p[0])
            ra = g.addLine(p[2], p[6])
            top = g.addLine(p[6], p[7])
            la = g.addLine(p[7], p[5])
            sub = g.addPlaneSurface([g.addCurveLoop([bot, rs, i_r, trace, i_l, ls_])])
            air = g.addPlaneSurface([g.addCurveLoop([-i_l, -trace, -i_r, ra, top, la])])
            g.synchronize()
            gmsh.model.addPhysicalGroup(2, [sub], dielectric_region(0))
            gmsh.model.addPhysicalGroup(2, [air], AIR)
            gmsh.model.addPhysicalGroup(1, [bot, trace, rs, ls_, ra, top, la], PEC)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _mode_problem(tmp_path, msh, *, eps_r=None, direction="z", freq=1e9):
    """Write the mode .pro for a hand-built cross-section and pair it with a setup."""
    from simpleEMS import fem_formulation
    from simpleEMS.fem_backend import FEMOptions, PortSpec, Problem, SolidSpec

    solids = {"port_1": SolidSpec("port_1", "port")}
    diel_regions = {}
    if eps_r is not None:
        solids["substrate"] = SolidSpec(
            "substrate", "dielectric", Dielectric(eps_r, 0.0)
        )
        diel_regions["substrate"] = dielectric_region(0)

    problem = Problem(
        step_file="s.step",
        name="sec",
        solids=solids,
        ports=[PortSpec("port_1", 1, 50.0, direction, "wave", "y")],
        freqs=np.array([freq]),
        options=FEMOptions(),
    )
    mesh = Mesh(
        msh_path=str(msh),
        dielectric_regions=diel_regions,
        air_region=AIR,
        pec_region=PEC,
        port_regions={
            1: PortMesh(
                number=1,
                region=port_region(1),
                direction=direction,
                z0=50.0,
                gap=1.6e-3,
                width=3e-3,
                center=(0, 0, 0),
                kind="wave",
                prop_dir="y",
            )
        },
        abc_region=ABC,
        boundary="silver_muller",
        bbox=(0, 0, 0, 1, 1, 1),
        box_bbox=(0, 0, 0, 1, 1, 1),
    )
    pro = fem_formulation.write_mode_problem(problem, mesh, 1, tmp_path)
    return pro


@pytest.mark.slow
@pytest.mark.needs_csxcad
@pytest.mark.needs_getdp_bin
class TestAgainstClosedForm:
    """The mode solve, checked where the answer is known exactly."""

    def _beta(self, tmp_path, msh, pro, freq, eps_max, bounds, direction="z"):
        setup = PortModeSetup(
            number=1,
            pro_path=pro,
            msh_path=str(msh),
            direction=direction,
            prop_axis=1,
            plane_at=0.0,
            axes=(2, 0),
            bounds=bounds,
            eps_max=eps_max,
            z0=50.0,
        )
        return fem_port_mode.solve_port_mode(setup, freq, tmp_path)

    def test_rectangular_waveguide_te10(self, tmp_path):
        # WR-90 at 10 GHz: beta = sqrt(k0^2 - (pi/a)^2), a textbook result
        a, b, freq = 22.86e-3, 10.16e-3, 10e9
        msh = tmp_path / "wr90.msh"
        _rect_section(msh, a, b, lc=a / 20)
        pro = _mode_problem(tmp_path, msh, freq=freq)
        mode = self._beta(tmp_path, msh, pro, freq, 1.0, (0.0, 0.0, a, b))

        k0 = 2 * math.pi * freq / C0
        expected = math.sqrt(k0**2 - (math.pi / a) ** 2)
        assert mode.beta.real == pytest.approx(expected, rel=2e-3)

    def test_parallel_plate_carries_a_tem_mode_at_exactly_k0(self, tmp_path):
        # The TEM case is the one microstrip and CPW live in, so it is the one
        # that has to be exact rather than merely close.
        w, h, freq = 20e-3, 5e-3, 10e9
        msh = tmp_path / "pp.msh"
        _rect_section(msh, w, h, walls="plates", lc=h / 6)
        pro = _mode_problem(tmp_path, msh, freq=freq)
        mode = self._beta(tmp_path, msh, pro, freq, 1.0, (0.0, 0.0, w, h))

        k0 = 2 * math.pi * freq / C0
        assert mode.beta.real == pytest.approx(k0, rel=1e-6)
        assert mode.n_eff.real == pytest.approx(1.0, rel=1e-6)

    def test_microstrip_effective_permittivity_matches_the_closed_form(self, tmp_path):
        from simpleEMS.calc import microstrip_impedance

        w, h_sub, trace_w, freq = 30e-3, 1.6e-3, 3.0e-3, 2.45e9
        msh = tmp_path / "ms.msh"
        _rect_section(
            msh,
            w,
            h_sub + 12e-3,
            split_at=(h_sub, trace_w),
            lc=w / 25,
        )
        pro = _mode_problem(tmp_path, msh, eps_r=4.4, freq=freq)
        mode = self._beta(tmp_path, msh, pro, freq, 4.4, (0.0, 0.0, w, h_sub + 12e-3))

        _z0, eps_eff = microstrip_impedance(trace_w * 1e3, h_sub * 1e3, 0.0, 4.4, freq)
        # Hammerstad-Jensen is itself only good to ~1%, and the mode box is
        # finite, so a few percent is the right tolerance here.
        assert mode.eps_eff.real == pytest.approx(eps_eff, rel=0.05)
        assert 1.0 < mode.n_eff.real < math.sqrt(4.4)

    def test_the_mode_is_written_where_the_3d_solve_will_look_for_it(self, tmp_path):
        a, b, freq = 22.86e-3, 10.16e-3, 10e9
        msh = tmp_path / "wr90.msh"
        _rect_section(msh, a, b, lc=a / 16)
        pro = _mode_problem(tmp_path, msh, freq=freq)
        mode = self._beta(tmp_path, msh, pro, freq, 1.0, (0.0, 0.0, a, b))

        view = Path(mode.pos_path)
        assert view == tmp_path / "output" / "mode_1.pos"
        assert view.read_text().startswith('View "mode_1" {')

    def test_the_mode_carries_one_watt(self, tmp_path):
        # S-parameters do not depend on the normalisation, but Zc and the
        # reported V/I do, so it has to actually hold.
        a, b, freq = 22.86e-3, 10.16e-3, 10e9
        msh = tmp_path / "wr90.msh"
        _rect_section(msh, a, b, lc=a / 16)
        pro = _mode_problem(tmp_path, msh, freq=freq)
        mode = self._beta(tmp_path, msh, pro, freq, 1.0, (0.0, 0.0, a, b))

        coords, steps = read_pos_steps(mode.pos_path)
        values = steps[:, 0, :, :] + 1j * steps[:, 1, :, :]
        # the view is in 3D coords; its area element is unchanged by the rigid
        # transform, so recompute in the plane the mode was solved on
        local = np.stack([coords[:, :, 2], coords[:, :, 0]], axis=-1)
        p0, p1, p2 = local[:, 0], local[:, 1], local[:, 2]
        areas = 0.5 * np.abs(
            (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        )
        from simpleEMS.fem_materials import MU0

        omega = 2 * math.pi * freq
        power = mode.beta.real / (2 * omega * MU0) * _integrate_sq(values, areas)
        assert power == pytest.approx(1.0, rel=1e-6)


@pytest.mark.needs_csxcad
@pytest.mark.needs_getdp_bin
def test_getdp_is_never_left_waiting_on_stdin(tmp_path):
    """The eigenvalue solver prompts for Arpack settings when it has no
    eigen.par; an inherited stdin that never delivers a line hangs the sweep."""
    from simpleEMS import fem_solver

    a, b, freq = 22.86e-3, 10.16e-3, 10e9
    msh = tmp_path / "wr90.msh"
    _rect_section(msh, a, b, lc=a / 10)
    pro = _mode_problem(tmp_path, msh, freq=freq)
    (tmp_path / "eigen.par").unlink(missing_ok=True)

    proc = subprocess.run(
        [
            fem_solver.find_getdp(),
            str(pro),
            "-msh",
            str(msh),
            "-setnumber",
            "FREQ",
            str(freq),
            "-setnumber",
            "SHIFT_RE",
            "25000",
            "-solve",
            "ModeAnalysis",
            "-v",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


@pytest.mark.needs_csxcad
class TestLossTangentRoundTrip:
    """kappa is fixed at one frequency; reading it back elsewhere scales loss."""

    @staticmethod
    def _csx(kappa: float):
        CSXCAD = pytest.importorskip("CSXCAD")
        csx = CSXCAD.ContinuousStructure()
        sub = csx.AddMaterial("substrate")
        sub.SetMaterialProperty(epsilon=4.4, kappa=kappa)
        sub.AddBox([-10, -15, 0], [10, 15, 1.6])
        return csx

    @staticmethod
    def _kappa(tand, freq, eps_r=4.4):
        from simpleEMS.fem_materials import EPS0

        return tand * 2 * math.pi * freq * EPS0 * eps_r

    def test_the_defining_frequency_recovers_the_loss_tangent(self):
        from simpleEMS.fem_backend import _csx_roles

        design = 2.45e9
        csx = self._csx(self._kappa(0.001, design))

        _roles, diels, _ports, _sigma = _csx_roles(csx, design)

        assert diels["substrate"].tan_d == pytest.approx(0.001, rel=1e-9)

    def test_another_frequency_scales_it_by_the_ratio(self):
        """This is what a sweep's band centre used to do to a 2.45 GHz board."""
        from simpleEMS.fem_backend import _csx_roles

        csx = self._csx(self._kappa(0.001, 2.45e9))

        _roles, diels, _ports, _sigma = _csx_roles(csx, 2.0e9)

        assert diels["substrate"].tan_d == pytest.approx(0.001 * 2.45 / 2.0, rel=1e-9)

    def test_a_lossless_board_stays_lossless(self):
        from simpleEMS.fem_backend import _csx_roles

        _roles, diels, _ports, _sigma = _csx_roles(self._csx(0.0), 2.45e9)

        assert diels["substrate"].tan_d == 0.0


class TestModeAttenuation:
    """A lossy mode's beta is complex; its imaginary part is the attenuation."""

    @staticmethod
    def _mode(beta):
        return fem_port_mode.PortMode(
            number=1, freq=2.45e9, beta=beta, n_eff=1.0, zc=50.0, pos_path=""
        )

    def test_a_lossless_mode_reports_no_attenuation(self):
        assert self._mode(93.77 + 0j).alpha_db_per_m == 0.0

    def test_a_lossy_mode_reports_its_attenuation(self):
        """FR-4 at 2.45 GHz: Hammerstad-Jensen gives about 0.37 dB/m."""
        mode = self._mode(93.4719 - 0.042867j)

        assert mode.alpha_np_per_m == pytest.approx(0.042867)
        assert mode.alpha_db_per_m == pytest.approx(0.3723, abs=5e-4)

    def test_a_growing_mode_is_not_reported_as_gain(self):
        """Numerical noise can put a small positive imaginary part on beta."""
        assert self._mode(93.77 + 1e-9j).alpha_db_per_m == 0.0


@pytest.mark.needs_csxcad
class TestFEMOptionsValidation:
    @pytest.mark.parametrize(
        "boundary", ["silver_muller", "pml", "pec"], ids=["abc", "pml", "pec"]
    )
    def test_every_outer_boundary_is_accepted(self, boundary):
        from simpleEMS.fem_backend import FEMOptions

        assert FEMOptions(boundary=boundary).boundary == boundary

    def test_an_unknown_outer_boundary_is_rejected_by_name(self):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="'silver_muller', 'pml' or 'pec'"):
            FEMOptions(boundary="absorbing")

    def test_a_mode_index_beyond_the_modes_computed_is_rejected_up_front(self):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="out of reach"):
            FEMOptions(port_mode_modes=2, port_mode_index=4)

    def test_defaults_select_the_fundamental(self):
        from simpleEMS.fem_backend import FEMOptions

        opts = FEMOptions()
        assert opts.port_mode_index == 0
        assert opts.port_mode_modes == 6

    @pytest.mark.parametrize(
        "kwargs",
        [{"port_mode_modes": 0}, {"port_mode_index": -1}],
        ids=["no-modes", "negative-index"],
    )
    def test_nonsense_is_rejected(self, kwargs):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError):
            FEMOptions(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"waveport_width_mm": 0.0},
            {"waveport_width_mm": -1.0},
            {"waveport_height_mm": 0.0},
            {"waveport_height_mm": -1.0},
        ],
        ids=["zero-width", "negative-width", "zero-height", "negative-height"],
    )
    def test_a_nonpositive_waveport_size_is_rejected(self, kwargs):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="must be positive"):
            FEMOptions(**kwargs)

    def test_the_waveport_size_reaches_the_options_from_the_params(self):
        from simpleEMS.components import GenericParams

        params = GenericParams(
            freq_range=(2e9, 3e9),
            main_freq=2.45e9,
            substrate_eps_r=4.4,
            substrate_tand=0.001,
            substrate_thickness_mm=1.6,
            substrate_width_mm=20.0,
            substrate_length_mm=30.0,
            FEM_waveport_width_mm=12.0,
            FEM_waveport_height_mm=8.0,
        )
        assert params.fem_options.waveport_width_mm == 12.0
        assert params.fem_options.waveport_height_mm == 8.0


@pytest.mark.needs_csxcad
class TestPortTypeOption:
    def test_the_default_is_the_lumped_port_the_backend_always_had(self):
        from simpleEMS.fem_backend import FEMOptions

        assert FEMOptions().port_type == "lumpedport"

    @pytest.mark.parametrize(
        "value", ["waveport", "lumpedport"], ids=["wave", "lumped"]
    )
    def test_both_choices_are_accepted(self, value):
        from simpleEMS.fem_backend import FEMOptions

        assert FEMOptions(port_type=value).port_type == value

    def test_a_typo_is_rejected_rather_than_silently_meaning_lumped(self):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="port_type must be"):
            FEMOptions(port_type="wave")

    def test_the_params_field_is_validated_at_construction(self):
        # not only when fem_options is built, so a typo surfaces immediately
        from simpleEMS.components import GenericParams

        kw = dict(
            freq_range=(2e9, 3e9),
            main_freq=2.45e9,
            substrate_eps_r=4.4,
            substrate_tand=0.001,
            substrate_thickness_mm=1.6,
            substrate_width_mm=20.0,
            substrate_length_mm=30.0,
        )
        with pytest.raises(ValueError, match="FEM_port_type must be"):
            GenericParams(**kw, FEM_port_type="wave")
        assert (
            GenericParams(**kw, FEM_port_type="waveport").fem_options.port_type
            == "waveport"
        )


class TestImpedanceOverride:
    def test_the_override_survives_the_trip_through_json(self):
        setup = make_setup(zc_override=50.0, target_eps_eff=2.9)
        again = PortModeSetup.from_dict(setup.to_dict())
        assert again.zc_override == 50.0
        assert again.target_eps_eff == 2.9

    def test_none_stays_none(self):
        assert PortModeSetup.from_dict(make_setup().to_dict()).zc_override is None

    @pytest.mark.needs_csxcad
    def test_a_nonpositive_override_is_rejected(self):
        from simpleEMS.fem_backend import FEMOptions

        with pytest.raises(ValueError, match="port_mode_zc must be positive"):
            FEMOptions(port_mode_zc=0.0)


class TestPropagationAxis:
    """Which axis the line runs along, worked out from the port's own extents."""

    # a board in the xy plane: thin in z, so z is the substrate normal
    BOARD = (-0.012, -0.015, 0.0, 0.012, 0.015, 0.0016)

    def test_a_lumped_port_box_is_flat_along_one_axis_and_that_settles_it(self):
        from simpleEMS.fem_geometry import _port_prop_axis

        # microstrip port: spans the trace width in x and the gap in z, flat in y
        bb = (-0.0015, -0.015, 0.0, 0.0015, -0.015, 0.0016)
        assert _port_prop_axis(bb, self.BOARD) == 1

    def test_a_cpw_gap_box_is_flat_along_two_and_the_board_normal_breaks_the_tie(
        self,
    ):
        from simpleEMS.fem_geometry import _port_prop_axis

        # CPW gap: spans the gap in x only; flat in y (the run) and z (the
        # board normal). Only the board normal tells them apart.
        bb = (0.001, -0.01439, 0.0016, 0.001445, -0.01439, 0.0016)
        assert _port_prop_axis(bb, self.BOARD) == 1

    def test_the_1e_3_export_floor_still_counts_as_flat(self):
        from simpleEMS.fem_geometry import _port_prop_axis

        # export_cad floors every box dimension at 1e-3 drawing units, so a
        # port drawn flat arrives 1 um thick, not 0
        bb = (-0.0015, -0.015, 0.0, 0.0015, -0.015 + 1e-6, 0.0016)
        assert _port_prop_axis(bb, self.BOARD) == 1

    def test_a_port_flat_along_nothing_is_refused_rather_than_guessed(self):
        from simpleEMS.fem_geometry import _port_prop_axis

        bb = (0.0, 0.0, 0.0, 0.01, 0.01, 0.01)
        with pytest.raises(RuntimeError, match="which axis this port's line"):
            _port_prop_axis(bb, self.BOARD)

    def test_a_probe_feed_flat_along_two_axes_including_the_normal_is_refused(
        self,
    ):
        from simpleEMS.fem_geometry import _port_prop_axis

        # a vertical probe is flat in x and y; neither is the board normal, so
        # there is no line direction to find and it says so
        bb = (0.001, 0.002, 0.0, 0.001, 0.002, 0.0016)
        with pytest.raises(RuntimeError, match="which axis this port's line"):
            _port_prop_axis(bb, self.BOARD)


class TestWavePortSpan:
    """A wave port's cross-section, sized around the line it terminates."""

    # a 1.6 mm board in the xy plane, the line running along y
    BOARD = (-0.012, -0.015, 0.0, 0.012, 0.015, 0.0016)
    DOMAIN = (-0.03, -0.035, -0.02, 0.03, 0.035, 0.02)
    PLANE = -0.015

    def port(self, width, x_centre=0.0):
        # a microstrip lumped port: the trace width in x, the gap in z
        return (
            x_centre - width / 2,
            self.PLANE,
            0.0,
            x_centre + width / 2,
            self.PLANE,
            0.0016,
        )

    def span(self, port_bbox, **kwargs):
        from simpleEMS.fem_geometry import _wave_port_span

        return _wave_port_span(port_bbox, 1, self.BOARD, self.DOMAIN, **kwargs)

    @pytest.mark.parametrize(
        ("trace_width", "port_width"),
        [(0.001, 0.010), (0.003, 0.015)],
        ids=["narrow-trace-ten-widths", "wide-trace-five-widths"],
    )
    def test_the_default_width_follows_the_trace(self, trace_width, port_width):
        span = self.span(self.port(trace_width))
        assert span[3] - span[0] == pytest.approx(port_width)
        assert 0.5 * (span[0] + span[3]) == pytest.approx(0.0)

    def test_the_default_height_is_six_substrates_from_the_ground(self):
        span = self.span(self.port(0.003))
        assert span[2] == pytest.approx(0.0)
        assert span[5] == pytest.approx(6 * 0.0016)

    def test_explicit_sizes_win_over_the_defaults(self):
        span = self.span(self.port(0.003), width_mm=12.0, height_mm=5.0)
        assert span[3] - span[0] == pytest.approx(0.012)
        assert span[5] - span[2] == pytest.approx(0.005)

    def test_an_oversized_port_is_clipped_to_the_domain(self):
        span = self.span(self.port(0.003), width_mm=500.0, height_mm=500.0)
        assert (span[0], span[3]) == pytest.approx((self.DOMAIN[0], self.DOMAIN[3]))
        assert span[5] == pytest.approx(self.DOMAIN[5])
        # still standing on the ground, not reaching into the air below it
        assert span[2] == pytest.approx(0.0)

    def test_the_plane_stays_where_the_port_was_drawn(self):
        span = self.span(self.port(0.003))
        assert (span[1], span[4]) == (self.PLANE, self.PLANE)

    def test_the_port_centres_on_a_line_off_the_board_centre(self):
        span = self.span(self.port(0.003, x_centre=0.004))
        assert 0.5 * (span[0] + span[3]) == pytest.approx(0.004)

    def test_a_cpw_port_is_sized_from_both_gaps_together(self):
        # two gap boxes, merged, span trace plus gaps: 3 mm here, on the surface
        merged = (-0.0015, self.PLANE, 0.0016, 0.0015, self.PLANE, 0.0016)
        span = self.span(merged)
        assert span[3] - span[0] == pytest.approx(0.015)
        assert span[2] == pytest.approx(0.0)

    def test_a_trace_under_the_substrate_grows_the_port_downwards(self):
        port_bbox = (-0.0015, self.PLANE, 0.0, 0.0015, self.PLANE, 0.0)
        span = self.span(port_bbox)
        assert span[5] == pytest.approx(0.0016)
        assert span[2] == pytest.approx(0.0016 - 6 * 0.0016)

    def test_a_line_along_the_board_normal_is_refused(self):
        from simpleEMS.fem_geometry import _wave_port_span

        with pytest.raises(RuntimeError, match="board normal"):
            _wave_port_span(self.port(0.003), 2, self.BOARD, self.DOMAIN)


@pytest.mark.needs_csxcad
class TestWavePortFaces:
    """A wave port stands on an end of the structure, and names that face.

    A wave port terminates the simulation domain, so the air box must not pad
    the face it sits on. These cover which face each port is found on, and the
    snap that absorbs the fraction of a millimetre openEMS's grid moves a CPW
    port's reference plane by.
    """

    # a 1.6 mm board in the xy plane, the line running along y
    BOARD = (-0.012, -0.015, 0.0, 0.012, 0.015, 0.0016)
    # FR-4 at 3 GHz: 47.6 mm, so ports may be snapped by up to 0.476 mm
    LAMBDA_MIN = 0.0476

    def sheet(self, at, width=0.003, x_centre=0.0):
        """A port sheet flat in y at ``at``, spanning the substrate in z."""
        return (x_centre - width / 2, at, 0.0, x_centre + width / 2, at, 0.001635)

    # (role, name, bbox, bbox_volume) as _snapshot_solids records them: the
    # board and its ground, plus a port solid the exporter gave a micron of
    # thickness, so it reaches half a micron past the board
    ORIGINALS = [
        ("dielectric", "substrate", (-0.012, -0.015, 0.0, 0.012, 0.015, 0.0016), 1.0),
        ("pec", "ground", (-0.012, -0.015, -3.5e-5, 0.012, 0.015, 0.0), 1.0),
        ("port", "p1", (-0.0015, -0.0150005, 0.0, 0.0015, -0.0149995, 0.001635), 0.0),
    ]

    def faces(self, port_geo, ports, originals=None):
        from simpleEMS.fem_backend import Problem
        from simpleEMS.fem_geometry import _wave_port_faces

        prob = Problem(step_file="s.step")
        prob.ports = ports
        return _wave_port_faces(
            prob,
            port_geo,
            self.ORIGINALS if originals is None else originals,
            self.BOARD,
            self.LAMBDA_MIN,
        )

    def wave_port(self, number, solids, prop_dir=""):
        from simpleEMS.fem_backend import PortSpec

        return PortSpec(
            solids[0],
            number,
            kind="wave",
            direction="z",
            prop_dir=prop_dir,
            solids=list(solids),
        )

    @pytest.mark.parametrize(
        ("at", "face"),
        [(-0.015, (1, 0)), (0.015, (1, 1))],
        ids=["low", "high"],
    )
    def test_a_flush_port_names_the_face_it_stands_on(self, at, face):
        faces, planes = self.faces({"p1": self.sheet(at)}, [self.wave_port(1, ["p1"])])

        assert set(faces) == {face}
        assert faces[face] == pytest.approx(at)
        assert planes == {1: pytest.approx(at)}

    def test_both_ports_of_a_line_terminate_their_own_face(self):
        faces, planes = self.faces(
            {"p1": self.sheet(-0.015), "p2": self.sheet(0.015)},
            [self.wave_port(1, ["p1"]), self.wave_port(2, ["p2"])],
        )

        assert faces == {
            (1, 0): pytest.approx(-0.015),
            (1, 1): pytest.approx(0.015),
        }
        assert planes == {1: pytest.approx(-0.015), 2: pytest.approx(0.015)}

    def test_a_port_snapped_off_the_end_by_the_grid_is_pulled_back(self):
        """openEMS puts a CPW port's reference plane on an FDTD grid line,
        which lands it a fraction of a millimetre inside the board."""
        faces, planes = self.faces(
            {"p1": self.sheet(-0.0149333)}, [self.wave_port(1, ["p1"])]
        )

        assert set(faces) == {(1, 0)}
        assert planes[1] == pytest.approx(-0.015)

    def test_a_port_in_the_middle_of_the_board_is_refused(self):
        with pytest.raises(RuntimeError, match="wave port 1 sits"):
            self.faces({"p1": self.sheet(0.0)}, [self.wave_port(1, ["p1"])])

    def test_the_refusal_names_both_ends_and_the_way_out(self):
        with pytest.raises(RuntimeError) as excinfo:
            self.faces({"p1": self.sheet(0.0)}, [self.wave_port(1, ["p1"])])

        message = str(excinfo.value)
        assert "-15.0000" in message and "15.0000" in message
        assert "lumpedport" in message

    def test_a_cpw_port_is_one_face_from_both_its_gap_solids(self):
        """Both slots are drawn as separate solids on the same plane."""
        faces, planes = self.faces(
            {
                "g1": self.sheet(-0.015, width=0.0005, x_centre=-0.0012),
                "g2": self.sheet(-0.015, width=0.0005, x_centre=0.0012),
            },
            [self.wave_port(1, ["g1", "g2"])],
        )

        assert set(faces) == {(1, 0)}
        assert planes == {1: pytest.approx(-0.015)}

    def test_an_explicit_prop_dir_is_honoured(self):
        """The line runs along x here, so the port terminates an x face."""
        port = (-0.012, -0.004, 0.0, -0.012, 0.004, 0.001635)
        faces, planes = self.faces(
            {"p1": port}, [self.wave_port(1, ["p1"], prop_dir="x")]
        )

        assert set(faces) == {(0, 0)}
        assert planes == {1: pytest.approx(-0.012)}

    def test_a_lumped_port_terminates_nothing(self):
        from simpleEMS.fem_backend import PortSpec

        faces, planes = self.faces(
            {"p1": self.sheet(0.0)},
            [PortSpec("p1", 1, direction="z")],
        )

        assert faces == {}
        assert planes == {}


@pytest.mark.needs_csxcad
class TestPortSolidGrouping:
    """A CPW port is drawn as one solid per gap; both belong to one port."""

    def test_extra_solids_join_the_port_instead_of_being_dropped(self):
        from simpleEMS.fem_backend import Problem, _register_port

        prob = Problem(step_file="s.step")
        seen: set[int] = set()
        _register_port(prob, seen, "port_resist_1", 1, 50.0, "x", "wave")
        _register_port(prob, seen, "port_resist_1_1", 1, 50.0, "x", "wave")

        assert len(prob.ports) == 1
        assert prob.ports[0].solids == ["port_resist_1", "port_resist_1_1"]

    def test_a_repeated_solid_is_not_added_twice(self):
        from simpleEMS.fem_backend import Problem, _register_port

        prob = Problem(step_file="s.step")
        seen: set[int] = set()
        _register_port(prob, seen, "port_resist_1", 1, 50.0)
        _register_port(prob, seen, "port_resist_1", 1, 50.0)
        assert prob.ports[0].solids == ["port_resist_1"]

    def test_a_lumped_port_has_exactly_its_own_solid(self):
        from simpleEMS.fem_backend import PortSpec

        assert PortSpec("port_resist_1", 1).solids == ["port_resist_1"]

    def test_different_port_numbers_stay_separate(self):
        from simpleEMS.fem_backend import Problem, _register_port

        prob = Problem(step_file="s.step")
        seen: set[int] = set()
        _register_port(prob, seen, "port_resist_1", 1, 50.0)
        _register_port(prob, seen, "port_resist_2", 2, 50.0)
        assert [p.number for p in prob.ports] == [1, 2]
        assert prob.ports[0].solids == ["port_resist_1"]
        assert prob.ports[1].solids == ["port_resist_2"]


@pytest.mark.needs_csxcad
class TestPortKindFromGeometry:
    """What the FEM backend makes of the ports openEMS already drew.

    Nothing FEM-specific is authored: the kind comes off how many boxes the
    lumped-element property carries, which is the only trace of the difference
    between AddLumpedPort and AddCPWPort that CSXCAD keeps.
    """

    @staticmethod
    def _csx(n_boxes: int, resistance: float):
        CSXCAD = pytest.importorskip("CSXCAD")
        csx = CSXCAD.ContinuousStructure()
        sub = csx.AddMaterial("substrate")
        sub.SetMaterialProperty(epsilon=4.4, kappa=0.0)
        sub.AddBox([-10, -15, 0], [10, 15, 1.6])
        port = csx.AddLumpedElement("port_resist_1", ny=0, R=resistance)
        for i in range(n_boxes):
            port.AddBox([1 + 2 * i, -15, 1.6], [1.4 + 2 * i, -15, 1.6])
        return csx

    def test_one_box_is_a_lumped_port_by_default(self):
        from simpleEMS.fem_backend import _csx_roles

        _roles, _diel, ports, _sigma = _csx_roles(self._csx(1, 50.0), 2.45e9)
        z0, _dir, number, kind, is_cpw = ports["port_resist_1"]
        assert (kind, is_cpw, number) == ("lumped", False, 1)
        assert z0 == 50.0

    def test_one_box_becomes_a_wave_port_when_asked(self):
        from simpleEMS.fem_backend import _csx_roles

        _roles, _diel, ports, _sigma = _csx_roles(
            self._csx(1, 50.0), 2.45e9, "waveport"
        )
        assert ports["port_resist_1"][3] == "wave"

    def test_two_boxes_are_a_cpw_port_and_a_wave_port_even_by_default(self):
        # a lumped sheet cannot represent the odd CPW mode, so this is not
        # left to the default -- it would return plausible wrong numbers
        from simpleEMS.fem_backend import _csx_roles

        _roles, _diel, ports, _sigma = _csx_roles(self._csx(2, 100.0), 2.45e9)
        _z0, _dir, _number, kind, is_cpw = ports["port_resist_1"]
        assert (kind, is_cpw) == ("wave", True)

    def test_a_cpw_ports_doubled_resistance_is_halved_back(self):
        # openEMS applies Feed_R "to each gap as 2*R", so 100 ohm on the
        # property is a 50 ohm line
        from simpleEMS.fem_backend import _csx_roles

        _roles, _diel, ports, _sigma = _csx_roles(self._csx(2, 100.0), 2.45e9)
        assert ports["port_resist_1"][0] == 50.0

    def test_a_lumped_ports_resistance_is_left_alone(self):
        from simpleEMS.fem_backend import _csx_roles

        _roles, _diel, ports, _sigma = _csx_roles(self._csx(1, 100.0), 2.45e9)
        assert ports["port_resist_1"][0] == 100.0
